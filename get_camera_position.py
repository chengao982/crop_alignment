import math

import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import json
import pycolmap
import read_write_model
import reconstructor
import GCP_evaluation as GCP

from Hierarchical_Localization.hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_covisibility, pairs_from_poses, \
    logger
from Hierarchical_Localization.hloc import triangulation, visualization
from Hierarchical_Localization.hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from Hierarchical_Localization.hloc.utils import viz_3d, viz
from Hierarchical_Localization.hloc.utils.io import get_keypoints, get_matches, read_image

from read_write_model import read_images_binary
from collections import defaultdict
from typing import List
import random
import cv2

# function to get nearest neighbors of imgs_to_add images
def get_pairs(model, imgs_to_add, output, num_matched):
    logger.info('Reading the COLMAP model...')
    images = read_images_binary(model / 'images.bin')

    logger.info(
        f'Computing pairs for {len(images)} reconstruction images and {len(imgs_to_add)} images to add ...')

    pairs_total = []
    for key in imgs_to_add.keys():
        images.update({-1: imgs_to_add[key]})

        ids, dist, dR = pairs_from_poses.get_pairwise_distances(images)
        scores = -dist

        invalid = np.full(dR.shape, True)
        invalid[dR.shape[0] - 1] = np.full(dR.shape[1], False)
        invalid[dR.shape[0] - 1][dR.shape[1] - 1] = True

        np.fill_diagonal(invalid, True)
        pairs = pairs_from_poses.pairs_from_score_matrix(scores, invalid, num_matched)
        pairs = [(images[ids[i]].name, images[ids[j]].name) for i, j in pairs]
        for pair in pairs:
            pairs_total.append(pair)

    logger.info(f'Found {len(pairs_total)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join(p) for p in pairs_total))

# pick 4 matches and create a colored plot
def color_matches(image_dir, query_name, loc, reconstruction=None,
                  db_image_dir=None, top_k_db=2, dpi=75):
    q_image = read_image(image_dir / query_name)
    if loc.get('covisibility_clustering', False):
        # select the first, largest cluster if the localization failed
        loc = loc['log_clusters'][loc['best_cluster'] or 0]

    inliers = np.array(loc['PnP_ret']['inliers'])
    mkp_q = loc['keypoints_query']
    n = len(loc['db'])
    if reconstruction is not None:
        # for each pair of query keypoint and its matched 3D point,
        # we need to find its corresponding keypoint in each database image
        # that observes it. We also count the number of inliers in each.
        kp_idxs, kp_to_3D_to_db = loc['keypoint_index_to_db']
        counts = np.zeros(n)
        dbs_kp_q_db = [[] for _ in range(n)]
        inliers_dbs = [[] for _ in range(n)]
        for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers,
                                                         kp_to_3D_to_db)):
            track = reconstruction.points3D[p3D_id].track
            track = {el.image_id: el.point2D_idx for el in track.elements}
            for db_idx in db_idxs:
                counts[db_idx] += inl
                kp_db = track[loc['db'][db_idx]]
                dbs_kp_q_db[db_idx].append((i, kp_db))
                inliers_dbs[db_idx].append(inl)
    else:
        # for inloc the database keypoints are already in the logs
        assert 'keypoints_db' in loc
        assert 'indices_db' in loc
        counts = np.array([
            np.sum(loc['indices_db'][inliers] == i) for i in range(n)])

    # display the database images with the most inlier matches
    db_sort = np.argsort(-counts)
    for db_idx in db_sort[:top_k_db]:
        if reconstruction is not None:
            db = reconstruction.images[loc['db'][db_idx]]
            db_name = db.name
            db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
            kp_q = mkp_q[db_kp_q_db[:, 0]]
            kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
            inliers_db = inliers_dbs[db_idx]
        else:
            db_name = loc['db'][db_idx]
            kp_q = mkp_q[loc['indices_db'] == db_idx]
            kp_db = loc['keypoints_db'][loc['indices_db'] == db_idx]
            inliers_db = inliers[loc['indices_db'] == db_idx]

        db_image = read_image((db_image_dir or image_dir) / db_name)

        random_idxs = random.choices(range(len(kp_q)), k=4)
        kp_q = np.array([kp_q[i] for i in random_idxs])
        kp_db = np.array([kp_db[i] for i in random_idxs])
        color = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.65, 0.0]]
        viz.plot_images([q_image, db_image], dpi=dpi)
        viz.plot_matches(kp_q, kp_db, color, a=0.8, ps=9, lw=2.5)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
        viz.add_text(0, query_name, **opts)
        viz.add_text(1, db_name, **opts)

# localize camera in model
def localize_cameras(ws_path, rec_path, temp_path, plotting, extractor, matcher):
    # define paths and params
    feature_conf = extract_features.confs[extractor]
    if 'weights_path' in feature_conf['model'].keys():
        feature_conf['model']['weights_path'] = os.path.dirname(__file__) + feature_conf['model']['weights_path']
    matcher_conf = match_features.confs[matcher]
    if 'weights_path' in matcher_conf['model'].keys():
        matcher_conf['model']['weights_path'] = os.path.dirname(__file__) + matcher_conf['model']['weights_path']
    number_of_neighbors = 4
    query_path = Path(temp_path + '/images')
    query = sorted([f for f in os.listdir(query_path) if os.path.isfile(os.path.join(query_path, f))])
    images = Path(rec_path + '/images')
    references = sorted([f for f in os.listdir(images) if os.path.isfile(os.path.join(images, f))])
    outputs = Path(ws_path + '/data')
    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'
    plot_directory = os.path.join(ws_path, 'plots')
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # reload existing colmap models
    temp_model = pycolmap.Reconstruction(temp_path + '/sparse/aligned')
    camera = temp_model.cameras[1]

    # prepare triangulated model
    # extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    # pairs_from_covisibility.main(Path(rec_path + '/sparse/aligned'), sfm_pairs, num_matched=5)
    # sfm_matches = match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    # reconstruction = triangulation.main(
    #     outputs / 'sift',
    #     Path(rec_path + '/sparse/aligned'),
    #     images,
    #     sfm_pairs,
    #     features,
    #     sfm_matches,)

    # get features, pairs and matches to localize images in model
    extract_features.main(feature_conf, query_path, image_list=query, feature_path=features, overwrite=True)
    images_to_add = read_images_binary(temp_path + '/sparse/aligned/images.bin')
    get_pairs(Path(rec_path + '/sparse/aligned'), images_to_add, loc_pairs, number_of_neighbors)
    match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True);
    ref_ids = []
    for r in references:
        try:
            ref_ids.append(reconstruction.find_image_with_name(r).image_id)
        except:
            pass
    conf = {
        'estimation': {'ransac': {'max_error': 12}}, #12
        'refinement': {'refine_focal_length': False, 'refine_extra_params': False},
    }
    # add base model to 3d plot
    if plotting:
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.5)', name="mapping")
    camera_locations_added = {}
    transformations = {}
    localizer = QueryLocalizer(reconstruction, conf)
    print(reconstruction)

    # localize query images q
    number_of_matches, number_of_inliers, inlier_ratios = np.empty((0, 1), float), np.empty((0, 1), float), np.empty((0, 1), float)
    for q in query:
        try:
            ret, log = pose_from_cluster(localizer, q, camera, ref_ids, features, matches)
            print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
            pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
            R = read_write_model.qvec2rotmat(ret['qvec'])
            Tr = ret['tvec']
            pos_add = np.matmul(-np.linalg.inv(R), np.array([[Tr[0]], [Tr[1]], [Tr[2]]]))
            camera_locations_added.update({q: [pos_add[0][0], pos_add[1][0], pos_add[2][0]]})
            transformations.update({q: [[R[0][0], R[0][1], R[0][2], Tr[0]], [R[1][0], R[1][1], R[1][2], Tr[1]],
                                        [R[2][0], R[2][1], R[2][2], Tr[2]], [0.0, 0.0, 0.0, 1.0]]})
            # add current camera to 3d plot and save matches as pdf
            if plotting:
                visualization.visualize_loc_from_log(images, query_path / q, log, reconstruction)
                viz.save_plot(plot_directory + '/' + q + '_query.pdf')
                plt.close()
                color_matches(images, query_path / q, log, reconstruction)
                viz.save_plot(plot_directory + '/' + q + '_color.pdf')
                plt.close()
                viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=q)
            inlier_ratios = np.append(inlier_ratios, ret["num_inliers"] / len(ret["inliers"]))
            number_of_matches = np.append(number_of_matches, log["num_matches"])
            number_of_inliers = np.append(number_of_inliers, ret["num_inliers"])
        except:
            inlier_ratios = np.append(inlier_ratios, 0.0)
            number_of_matches = np.append(number_of_matches, 0.0)
            number_of_inliers = np.append(number_of_inliers, 0.0)

    # save data
    with open(outputs / 'localization_data.json', 'w') as outfile:
        json.dump(camera_locations_added, outfile)
    with open(outputs / 'transformation_data.json', 'w') as outfile:
        json.dump(transformations, outfile)
    np.savetxt(outputs / 'number_matches.out', number_of_matches)
    np.savetxt(outputs / 'number_inliers.out', number_of_inliers)
    np.savetxt(outputs / 'inlier_ratios.out', inlier_ratios)

    # visualize pointcloud with added cameras
    if plotting:
        fig.show()


# compute affine transform from raw to corr frame for img with name
def get_cam_to_cam_transform(T_raw, T_corr, name):
    T_raw_cam = np.linalg.inv(T_raw[name])
    T_corr_cam = np.linalg.inv(T_corr[name])

    vec1 = np.array([[1.0], [0.0], [0.0], [1.0]])
    vec2 = np.array([[0.0], [1.0], [0.0], [1.0]])
    vec3 = np.array([[0.0], [0.0], [1.0], [1.0]])
    vec4 = np.array([[1.0], [1.0], [1.0], [1.0]])

    M_raw = np.column_stack((np.matmul(T_raw_cam, vec1), np.matmul(T_raw_cam, vec2),
                             np.matmul(T_raw_cam, vec3), np.matmul(T_raw_cam, vec4)))
    M_corr = np.column_stack((np.matmul(T_corr_cam, vec1), np.matmul(T_corr_cam, vec2),
                              np.matmul(T_corr_cam, vec3), np.matmul(T_corr_cam, vec4)))

    T = np.matmul(M_corr, np.linalg.inv(M_raw))
    return T


# load poses and transformations (if transformation_bool=True) before and after alignment
def load_data(raw_path, corrected_path, transformation_bool):
    images_raw = read_images_binary(raw_path + '/sparse/aligned/images.bin')
    raw_poses = {}
    for id in images_raw:
        R = images_raw[id].qvec2rotmat()
        pos = np.matmul(-np.linalg.inv(R), images_raw[id].tvec)
        raw_poses.update({images_raw[id].name: pos})
    raw_poses = dict(sorted(raw_poses.items()))

    with open(corrected_path + '/data/localization_data.json', "r") as infile:
        data = []
        for line in infile:
            data.append(json.loads(line))
    corr_poses = data[0]

    ground_truth = reconstructor.get_gps_poses(raw_path, 0.0)

    if transformation_bool == True:
        with open(corrected_path + '/data/transformation_data.json', "r") as infile:
            data = []
            for line in infile:
                data.append(json.loads(line))
        T_corr = data[0]

        T_raw = {}
        for key in T_corr:
            for id in images_raw:
                if images_raw[id].name == key:
                    R = images_raw[id].qvec2rotmat()
                    T_mat_raw = [[R[0][0], R[0][1], R[0][2], images_raw[id].tvec[0]],
                                 [R[1][0], R[1][1], R[1][2], images_raw[id].tvec[1]],
                                 [R[2][0], R[2][1], R[2][2], images_raw[id].tvec[2]],
                                 [0.0, 0.0, 0.0, 1.0]]
                    T_raw.update({key: T_mat_raw})
        T = {}
        for name in corr_poses:
            T.update({name: get_cam_to_cam_transform(T_raw, T_corr, name)})
        return raw_poses, corr_poses, ground_truth, T
    else:
        return raw_poses, corr_poses, ground_truth


# compute mean distance of two corresponding cameras in two lists of camera positions
def get_error_per_cam(transformed_points, i, j):
    errors = transformed_points[i] - transformed_points[j]
    total_dist = 0
    for e in errors:
        total_dist += np.linalg.norm(e)
    error_per_camera = total_dist / len(errors)
    return error_per_camera

# Remove cameras with errors bigger than gps noise and then find inliers and outliers according to transformation
# matrix. Create plots and save validated cameras as file
def filter_transformations(T, raw_poses, corr_poses, gt_poses, gps_noise, distance_threshold, workspace_path, plotting):
    errors_raw, errors_corr = [], []
    improved_cams = 0
    corr_poses_filtered, raw_poses_filtered, T_filtered = {}, {}, []
    for img in corr_poses:
        corr = corr_poses[img]
        raw = raw_poses[img]
        gt = gt_poses[img]
        errors_raw.append(np.linalg.norm(np.subtract([gt[0], gt[1], gt[2]], raw)))
        corr_error = np.linalg.norm(np.subtract([gt[0], gt[1], gt[2]], corr))
        if np.linalg.norm(np.subtract(raw, corr)) < gps_noise:
            errors_corr.append(corr_error)
            corr_poses_filtered.update({img: corr_poses[img]})
            raw_poses_filtered.update({img: raw_poses[img]})
            T_filtered.append(T[img])
            if errors_raw[-1] > corr_error:
                improved_cams += 1
        else:
            errors_corr.append(-0.05 * gps_noise)

    errors_corr_to_consider = [a for a in errors_corr if not a < 0.0]
    if not errors_corr_to_consider:
        error_text = "Position error \nbefore alignment: \nmean: " + str(round(np.mean(errors_raw), 5)) + \
                     "\nstd dev: " + str(round(np.std(errors_raw), 5)) + "\nAfter alignment: \nno cameras localized"
    else:
        error_text = "Position error \nbefore alignment: \nmean: " + str(round(np.mean(errors_raw), 5)) + \
                     "\nstd dev: " + str(round(np.std(errors_raw), 5)) + "\nAfter alignment: \nmean: " + \
                     str(round(np.mean(errors_corr_to_consider), 5)) + "\nstd dev: " + str(round(np.std(errors_corr_to_consider), 5))
    print(error_text)
    if plotting:
        plt.clf()
        fig = plt.figure()
        X = np.arange(len(errors_corr))
        a = plt.bar(X + 0.00, errors_raw, color='b', width=0.25)
        b = plt.bar(X + 0.25, errors_corr, color='g', width=0.25)
        plt.legend((a, b), ('error $C_1^r$', 'error $C_1^c$'),
                   loc='upper right', fontsize=9)
        plt.xticks(X + 0.125, '', fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylabel('Error in m', fontsize=9)
        plt.ylim([-0.025 * gps_noise, 1.5 * gps_noise])
        plt.title('Position errors in camera position')
        plt.figtext(0.125, 0.05, 'Number of improved cameras: ' + str(improved_cams) + '/' + str(len(errors_corr)))
        plt.tick_params(axis='x', which='both', bottom=False)
        plt.savefig(workspace_path + '/camera_errors.pdf')
        plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        inliers, outliers = compute_inlier(T_filtered, raw_poses_filtered, corr_poses_filtered, distance_threshold, workspace_path, figure=ax)

        raw_inlier, corr_inlier, gt_inlier = np.empty((0, 3), float), np.empty((0, 3), float), np.empty((0, 3), float)
        for name in inliers:
            raw_inlier = np.append(raw_inlier, [raw_poses[name]], axis=0)
            corr_inlier = np.append(corr_inlier, [corr_poses[name]], axis=0)
            gt_inlier = np.append(gt_inlier, [[gt_poses[name][0], gt_poses[name][1], gt_poses[name][2]]], axis=0)

        raw_outlier, corr_outlier, gt_outlier = np.empty((0, 3), float), np.empty((0, 3), float), np.empty((0, 3), float)
        for name in outliers:
            raw_outlier = np.append(raw_outlier, [raw_poses[name]], axis=0)
            corr_outlier = np.append(corr_outlier, [corr_poses[name]], axis=0)
            gt_outlier = np.append(gt_outlier, [[gt_poses[name][0], gt_poses[name][1], gt_poses[name][2]]], axis=0)

        x_corr_inlier, y_corr_inlier, z_corr_inlier = zip(*corr_inlier)
        x_gt_inlier, y_gt_inlier, z_gt_inlier = zip(*gt_inlier)
        x_raw_inlier, y_raw_inlier, z_raw_inlier = zip(*raw_inlier)
        x_corr_outlier, y_corr_outlier, z_corr_outlier = zip(*corr_outlier)
        x_gt_outlier, y_gt_outlier, z_gt_outlier = zip(*gt_outlier)
        x_raw_outlier, y_raw_outlier, z_raw_outlier = zip(*raw_outlier)

        error = []
        for i in range(len(x_corr_inlier)):
            error.append(math.sqrt((x_corr_inlier[i] - x_gt_inlier[i]) ** 2 +
                                   (y_corr_inlier[i] - y_gt_inlier[i]) ** 2 +
                                   (z_corr_inlier[i] - z_gt_inlier[i]) ** 2))
        error_text = error_text + "\n\nAfter validation: \nmean: " + str(round(np.mean(error), 5)) + \
                     "\nstd dev: " + str(round(np.std(error), 5)) + "\nInliers: " + \
                     str(len(x_corr_inlier)) + "/" + str(len(x_corr_outlier) + len(x_corr_inlier))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('\n\nX direction', fontsize=9)
        ax.set_ylabel('\n\nY direction', fontsize=9)
        ax.set_zlabel('\n\nZ direction', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        ax.xaxis.offsetText.set_fontsize(7)
        ax.yaxis.offsetText.set_fontsize(7)
        ax.set_title('Camera poses before and after temporal alignment')
        ax.scatter(x_raw_outlier, y_raw_outlier, z_raw_outlier, c='red', marker="+")
        ax.scatter(x_gt_outlier, y_gt_outlier, z_gt_outlier, c='blue', marker="+")
        ax.scatter(x_corr_outlier, y_corr_outlier, z_corr_outlier, c='green', marker="+")
        r = ax.scatter(x_raw_inlier, y_raw_inlier, z_raw_inlier, c='red')
        g = ax.scatter(x_gt_inlier, y_gt_inlier, z_gt_inlier, c='blue')
        c = ax.scatter(x_corr_inlier, y_corr_inlier, z_corr_inlier, c='green')
        plt.legend((r, c, g), ('Poses $C_1^r$', 'Poses $C_1^c$', 'Poses $C_1^{gt}$'), loc='upper right', fontsize=9)
        plt.axis('equal')
        plt.figtext(0.02, 0.35, error_text, fontsize=9)
        plt.savefig(workspace_path + '/camera_poses.pdf')
        plt.clf()

        print("plots created")

        with open(workspace_path + '/data/inlier_GPS.txt', 'w') as f:
            for img_name in inliers:
                coords = corr_poses[img_name]
                f.write(img_name + ' ' + str(coords[0]) + ' ' + str(coords[1]) + ' ' + str(coords[2]) + '\n')
        print("inlier_GPS.txt created in .../data/")

        return inliers, outliers

    else:
        inliers, outliers = compute_inlier(T_filtered, raw_poses_filtered, corr_poses_filtered, distance_threshold, workspace_path)

        with open(workspace_path + '/data/inlier_GPS.txt', 'w') as f:
            for img_name in inliers:
                coords = corr_poses[img_name]
                f.write(img_name + ' ' + str(coords[0]) + ' ' + str(coords[1]) + ' ' + str(coords[2]) + '\n')
        print("inlier_GPS.txt created in .../data/")

        return inliers, outliers

# Transform raw_poses with different T first. Then compute distances to other transformed points and
# get inliers and outliers
def compute_inlier(T, raw_poses, corr_poses, distance_threshold, workspace_path, figure=None):
    transformed_points = []
    for t in T:
        trans_poses = {}
        for key in raw_poses:
            transformed = np.matmul(t, np.array([[raw_poses[key][0]], [raw_poses[key][1]], [raw_poses[key][2]], [1.0]]))
            trans_poses.update({key: np.array([transformed[0][0], transformed[1][0], transformed[2][0]])})

        trans_list = np.empty((0, 3), float)
        for img in corr_poses:
            trans_list = np.append(trans_list, [trans_poses[img]], axis=0)
        transformed_points.append(trans_list)

    distance_mat = np.zeros((len(transformed_points), len(transformed_points)))
    for idx_i in range(len(transformed_points)):
        for idx_j in range(len(transformed_points)):
            if idx_j > idx_i:
                distance = get_error_per_cam(transformed_points, idx_i, idx_j)
                distance_mat[idx_i][idx_j] = distance
                distance_mat[idx_j][idx_i] = distance
    dist = []
    for i in range(len(distance_mat)):
        dist.append(sum(distance_mat[i]))
    center_idx = dist.index(min(dist))
    done = False
    while done == False:
        inliers = [center_idx]
        outliers = []
        for i in range(len(distance_mat)):
            if distance_mat[center_idx][i]:
                if distance_mat[center_idx][i] < distance_threshold:
                    inliers.append(i)
                else:
                    outliers.append(i)
        if len(inliers) > 3:
            done = True
        else:
            distance_threshold += 0.05

    if not figure is None:
        for o in outliers:
            x_trans_list, y_trans_list, z_trans_list = zip(*transformed_points[o])
            outl = figure.scatter(x_trans_list, y_trans_list, z_trans_list, color='black', marker="+", s=5)
        for i in inliers:
            x_trans_list, y_trans_list, z_trans_list = zip(*transformed_points[i])
            inl = figure.scatter(x_trans_list, y_trans_list, z_trans_list, color=[0.0, 1.0, 0.0], marker="+")

        figure.set_xlabel('\n\nX direction', fontsize=9)
        figure.set_ylabel('\n\nY direction', fontsize=9)
        figure.set_zlabel('\n\nZ direction', fontsize=9)
        figure.tick_params(axis='both', which='major', labelsize=7)
        figure.tick_params(axis='both', which='minor', labelsize=7)
        figure.xaxis.offsetText.set_fontsize(7)
        figure.yaxis.offsetText.set_fontsize(7)
        figure.set_title('Inliers and Outliers')
        plt.legend((inl, outl), ('inliers', 'outliers'), loc='upper left', fontsize=5)
        #plt.savefig(workspace_path + '/transformed_inliers.pdf')
        plt.clf()

    names_inlier = []
    for i in inliers:
        names_inlier.append(list(raw_poses.keys())[i])

    names_outlier = []
    for o in outliers:
        names_outlier.append(list(raw_poses.keys())[o])

    return names_inlier, names_outlier

# use colmaps model aligner to find similarity transform to align validated cameras
def correct_model(ws_path, model_path):
    reconstructor.align_with_gps(model_path, 'sparse/aligned', 'sparse/corrected',
                                 ws_path + '/data/inlier_GPS.txt', 'correction_output')

# extract features and localize cameras of temp model in gt model. Then validate the localization and
# align model with validated cameras
def main(workspace_path, reconstruction_gt_path, reconstruction_temp_path, plotting, gps_noise, dist_threshold, extractor, matcher):
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    localize_cameras(workspace_path, reconstruction_gt_path, reconstruction_temp_path, plotting, extractor, matcher)

    raw_poses, corr_poses, gt_poses, T = load_data(reconstruction_temp_path, workspace_path, True)
    inlier_list, outlier_list = filter_transformations(T, raw_poses, corr_poses, gt_poses, gps_noise, dist_threshold,
                                                       workspace_path, plotting)
    correct_model(workspace_path, reconstruction_temp_path)


if __name__ == "__main__":
    start_time = time.time()

    plot = False
    gps_error = 5.0
    distance_threshold = 0.20
    extractor = 'superpoint_custom'
    matcher = 'superglue'

    # TODO: adapt this path
    basedir = '/path/to/experiment'
    reconstruction_gt_path = basedir + '/20180322gt'
    reconstruction_temp_path = basedir + '/20180402'
    workspace_path = basedir + '/Superpoint_custom/20180402'
    main(workspace_path, reconstruction_gt_path, reconstruction_temp_path, plot, gps_error, distance_threshold, extractor, matcher)

    end_time = time.time()
    run_time = end_time - start_time
    print("Runtime: ", run_time)


