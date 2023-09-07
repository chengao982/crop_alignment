import os
import time
import numpy as np
import json
import cv2
import pycolmap
import shutil
import random
import read_write_model
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from read_write_model import read_images_binary
from collections import defaultdict
from pathlib import Path
from reconstructor import Reconstruction
from evaluator import Evaluation
from typing import List

from Hierarchical_Localization.hloc.utils.io import get_keypoints, get_matches, read_image
from Hierarchical_Localization.hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from Hierarchical_Localization.hloc import triangulation, visualization, logger
from Hierarchical_Localization.hloc import extract_features, match_features, match_dense, pairs_from_covisibility, pairs_from_exhaustive, pairs_from_poses
from Hierarchical_Localization.hloc.utils import viz_3d, viz

class CameraLocalization:
    def __init__(self, 
                 output_path, 
                 images_ref_path, 
                 images_temp_path, 
                 reconstruction_ref_path, 
                 reconstruction_temp_path, 
                 extractor, 
                 matcher,
                 plotting=False,
                 gps_noise=5.0, 
                 dist_threshold=0.20
                 ):
        self.output_path = output_path
        self.images_ref_path = images_ref_path
        self.images_temp_path = images_temp_path
        self.reconstruction_ref_path = reconstruction_ref_path
        self.reconstruction_temp_path = reconstruction_temp_path
        self.plotting = plotting
        self.gps_noise = gps_noise
        self.dist_threshold = dist_threshold
        self.extractor = extractor
        self.matcher = matcher
        self.is_successful = True

        images_ref_path_components = self.images_ref_path.split(os.path.sep)
        images_temp_path_components = self.images_temp_path.split(os.path.sep)
        self.images_ref_relative_path = os.path.sep.join(images_ref_path_components[-2:])
        self.images_temp_relative_path = os.path.sep.join(images_temp_path_components[-2:])
        self.images_base_path = os.path.sep.join(images_ref_path_components[:-2])

    # function to get nearest neighbors of imgs_to_add images
    def get_pairs(self, model, imgs_to_add, output, num_matched):
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

    # adaption of localize_sfm.pose_from_cluster: do not throw error when matching pair
    # does not exist in file
    def pose_from_cluster_try(self, localizer: QueryLocalizer, qname: str, query_camera: pycolmap.Camera,
                            db_ids: List[int], features_path: Path, matches_path: Path, **kwargs):
        kpq = get_keypoints(features_path, qname)
        kpq += 0.5  # COLMAP coordinates

        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        num_matches = 0
        for i, db_id in enumerate(db_ids):
            image = localizer.reconstruction.images[db_id]
            if image.num_points3D() == 0:
                logger.debug(f'No 3D points found for {image.name}.')
                continue
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                    for p in image.points2D])

            try:
                matches, _ = get_matches(matches_path, qname, image.name)
                matches = matches[points3D_ids[matches[:, 1]] != -1]
                num_matches += len(matches)
                for idx, m in matches:
                    id_3D = points3D_ids[m]
                    kp_idx_to_3D_to_db[idx][id_3D].append(i)
                    # avoid duplicate observations
                    if id_3D not in kp_idx_to_3D[idx]:
                        kp_idx_to_3D[idx].append(id_3D)
            except:
                pass

        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
        ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
        ret['camera'] = {
            'model': query_camera.model_name,
            'width': query_camera.width,
            'height': query_camera.height,
            'params': query_camera.params,
        }

        # mostly for logging and post-processing
        mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                        for i in idxs for j in kp_idx_to_3D[i]]
        log = {
            'db': db_ids,
            'PnP_ret': ret,
            'keypoints_query': kpq[mkp_idxs],
            'points3D_ids': mp3d_ids,
            'points3D_xyz': None,  # we don't log xyz anymore because of file size
            'num_matches': num_matches,
            'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
        }
        return ret, log

    # pick 4 matches and create a colored plot
    def color_matches(self, image_dir, query_name, loc, reconstruction=None,
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

    def save_3d_plot(self, fig, save_path):
        save_path = save_path + '.html'
        fig.write_html(save_path)

    # localize camera in model
    def localize_cameras(self):
        if self.extractor is not None and 'loftr' not in self.matcher:
            self._localize_cameras()
        elif self.extractor is None and 'loftr' in self.matcher:
            self._localize_cameras_loftr()
        else:
            raise Exception(f'extractor is None iff matcher is loftr.\nextractor:{self.extractor}, matcher:{self.matcher}')

    def _localize_cameras(self):
        # define paths and params
        feature_conf = extract_features.confs[self.extractor]
        matcher_conf = match_features.confs[self.matcher]
        number_of_neighbors = 10

        images = Path(self.images_base_path)
        references = [str(p.relative_to(images)) for p in sorted((Path(self.images_ref_path)).iterdir())]
        queries = [str(p.relative_to(images)) for p in sorted((Path(self.images_temp_path)).iterdir())]

        outputs = Path(self.output_path + '/data')
        # shutil.rmtree(self.output_path, ignore_errors=True)
        outputs.mkdir(parents=True, exist_ok=True)
        sfm_pairs = outputs / 'pairs-sfm.txt'
        loc_pairs = outputs / 'pairs-loc.txt'
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'
        plot_directory = os.path.join(self.output_path, 'plots')
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        # reload existing colmap models
        temp_model = pycolmap.Reconstruction(self.reconstruction_temp_path)
        camera = temp_model.cameras[1]

        extract_features.main(feature_conf, images, image_list=references, feature_path=features)
        pairs_from_covisibility.main(Path(self.reconstruction_ref_path), sfm_pairs, num_matched=5)
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        reconstruction = triangulation.main(
            outputs / 'sift',
            Path(self.reconstruction_ref_path),
            images,
            sfm_pairs,
            features,
            matches,
        )

        # add base model to 3d plot
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.5)', name="mapping")
        self.save_3d_plot(fig, os.path.join(plot_directory, 'ref_model'))


        # get features, pairs and matches to localize images in model
        extract_features.main(feature_conf, images, image_list=queries, feature_path=features)
        images_to_add = read_images_binary(os.path.join(self.reconstruction_temp_path, 'images.bin'))
        self.get_pairs(Path(self.reconstruction_ref_path), images_to_add, loc_pairs, number_of_neighbors)
        # references_registered = [reconstruction.images[i].name for i in reconstruction.reg_image_ids()]
        # pairs_from_exhaustive.main(loc_pairs, image_list=query, ref_list=references_registered)
        # ref_ids = [reconstruction.find_image_with_name(n).image_id for n in references_registered]
        match_features.main(matcher_conf, loc_pairs, features=features, matches=matches)
        ref_ids = []
        for r in references:
            try:
                ref_ids.append(reconstruction.find_image_with_name(r).image_id)
            except:
                pass

        conf = {
            'estimation': {'ransac': {'max_error': 12}},  # 12
            'refinement': {'refine_focal_length': False, 'refine_extra_params': False},
        }

        qvecs = {}
        camera_locations_added = {}
        transformations = {}
        localizer = QueryLocalizer(reconstruction, conf)
        print(reconstruction)

        # localize query images q
        number_of_matches, number_of_inliers, inlier_ratios = np.empty((0, 1), float), np.empty((0, 1), float), np.empty((0, 1), float)
        for q_id, q in enumerate(queries):
            try:
                q_path = q
                q = os.path.basename(q)
                ret, log = self.pose_from_cluster_try(localizer, q_path, camera, ref_ids, features, matches)
                print(f'{q}: found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
                assert ret["num_inliers"] >= 10, "Find less then 10 inliers"
                pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
                R = read_write_model.qvec2rotmat(ret['qvec'])
                Tr = ret['tvec']
                pos_add = np.matmul(-np.linalg.inv(R), np.array([[Tr[0]], [Tr[1]], [Tr[2]]]))
                qvecs.update({q: ret['qvec'].tolist()})
                camera_locations_added.update({q: [pos_add[0][0], pos_add[1][0], pos_add[2][0]]})
                transformations.update({q: [[R[0][0], R[0][1], R[0][2], Tr[0]], [R[1][0], R[1][1], R[1][2], Tr[1]],
                                            [R[2][0], R[2][1], R[2][2], Tr[2]], [0.0, 0.0, 0.0, 1.0]]})
                
                if self.plotting:
                    viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=q)
                    self.save_3d_plot(fig, os.path.join(plot_directory, 'localized_cameras'))
                    # if q_id % 8 == 0:
                    #     visualization.visualize_loc_from_log(images, q_path, log, reconstruction)
                    #     viz.save_plot(plot_directory + '/' + q + '_query.pdf')
                    #     plt.close('all')
                        # self.color_matches(images, q_path, log, reconstruction)
                        # viz.save_plot(plot_directory + '/' + q + '_color.pdf')
                        # plt.close('all')


                inlier_ratios = np.append(inlier_ratios, ret["num_inliers"] / len(ret["inliers"]))
                number_of_matches = np.append(number_of_matches, log["num_matches"])
                number_of_inliers = np.append(number_of_inliers, ret["num_inliers"])
            except:
                print(f'{q} localization failed')
                inlier_ratios = np.append(inlier_ratios, 0.0)
                number_of_matches = np.append(number_of_matches, 0.0)
                number_of_inliers = np.append(number_of_inliers, 0.0)

        # save data
        with open(outputs / 'qvec_data.json', 'w') as outfile:
            json.dump(qvecs, outfile)
        with open(outputs / 'localization_data.json', 'w') as outfile:
            json.dump(camera_locations_added, outfile)
        with open(outputs / 'transformation_data.json', 'w') as outfile:
            json.dump(transformations, outfile)
        np.savetxt(outputs / 'number_matches.out', number_of_matches)
        np.savetxt(outputs / 'number_inliers.out', number_of_inliers)
        np.savetxt(outputs / 'inlier_ratios.out', inlier_ratios)


    def _localize_cameras_loftr(self):
        # define paths and params
        matcher_conf = match_dense.confs[self.matcher]
        number_of_neighbors = 10

        images = Path(self.images_base_path)
        references = [str(p.relative_to(images)) for p in sorted((Path(self.images_ref_path)).iterdir())]
        queries = [str(p.relative_to(images)) for p in sorted((Path(self.images_temp_path)).iterdir())]

        outputs = Path(self.output_path + '/data')
        # shutil.rmtree(self.output_path, ignore_errors=True)
        outputs.mkdir(parents=True, exist_ok=True)
        sfm_pairs = outputs / 'pairs-sfm.txt'
        loc_pairs = outputs / 'pairs-loc.txt'
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'
        plot_directory = os.path.join(self.output_path, 'plots')
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        # reload existing colmap models
        temp_model = pycolmap.Reconstruction(self.reconstruction_temp_path)
        camera = temp_model.cameras[1]

        pairs_from_covisibility.main(Path(self.reconstruction_ref_path), sfm_pairs, num_matched=5)
        match_dense.main(matcher_conf, sfm_pairs, images, features=features, matches=matches)
        reconstruction = triangulation.main(
            outputs / 'sift',
            Path(self.reconstruction_ref_path),
            images,
            sfm_pairs,
            features,
            matches,
        )

        # add base model to 3d plot
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.5)', name="mapping")
        self.save_3d_plot(fig, os.path.join(plot_directory, 'ref_model'))


        # get features, pairs and matches to localize images in model
        images_to_add = read_images_binary(os.path.join(self.reconstruction_temp_path, 'images.bin'))
        self.get_pairs(Path(self.reconstruction_ref_path), images_to_add, loc_pairs, number_of_neighbors)
        # references_registered = [reconstruction.images[i].name for i in reconstruction.reg_image_ids()]
        # pairs_from_exhaustive.main(loc_pairs, image_list=query, ref_list=references_registered)
        # ref_ids = [reconstruction.find_image_with_name(n).image_id for n in references_registered]
        match_dense.main(matcher_conf, loc_pairs, images, outputs, 
                         matches=matches, features=features, max_kps=None)
        ref_ids = []
        for r in references:
            try:
                ref_ids.append(reconstruction.find_image_with_name(r).image_id)
            except:
                pass

        conf = {
            'estimation': {'ransac': {'max_error': 12}},  # 12
            'refinement': {'refine_focal_length': False, 'refine_extra_params': False},
        }

        qvecs = {}
        camera_locations_added = {}
        transformations = {}
        localizer = QueryLocalizer(reconstruction, conf)
        print(reconstruction)

        # localize query images q
        number_of_matches, number_of_inliers, inlier_ratios = np.empty((0, 1), float), np.empty((0, 1), float), np.empty((0, 1), float)
        for q_id, q in enumerate(queries):
            try:
                q_path = q
                q = os.path.basename(q)
                ret, log = self.pose_from_cluster_try(localizer, q_path, camera, ref_ids, features, matches)
                print(f'{q}: found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
                assert ret["num_inliers"] >= 10, "Find less then 10 inliers"
                pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
                R = read_write_model.qvec2rotmat(ret['qvec'])
                Tr = ret['tvec']
                pos_add = np.matmul(-np.linalg.inv(R), np.array([[Tr[0]], [Tr[1]], [Tr[2]]]))
                qvecs.update({q: ret['qvec'].tolist()})
                camera_locations_added.update({q: [pos_add[0][0], pos_add[1][0], pos_add[2][0]]})
                transformations.update({q: [[R[0][0], R[0][1], R[0][2], Tr[0]], [R[1][0], R[1][1], R[1][2], Tr[1]],
                                            [R[2][0], R[2][1], R[2][2], Tr[2]], [0.0, 0.0, 0.0, 1.0]]})
                
                if self.plotting:
                    viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=q)
                    self.save_3d_plot(fig, os.path.join(plot_directory, 'localized_cameras'))
                    # if q_id % 8 == 0:
                    #     visualization.visualize_loc_from_log(images, q_path, log, reconstruction)
                    #     viz.save_plot(plot_directory + '/' + q + '_query.pdf')
                    #     plt.close('all')
                        # self.color_matches(images, q_path, log, reconstruction)
                        # viz.save_plot(plot_directory + '/' + q + '_color.pdf')
                        # plt.close('all')

                inlier_ratios = np.append(inlier_ratios, ret["num_inliers"] / len(ret["inliers"]))
                number_of_matches = np.append(number_of_matches, log["num_matches"])
                number_of_inliers = np.append(number_of_inliers, ret["num_inliers"])
            except:
                print(f'{q} localization failed')
                inlier_ratios = np.append(inlier_ratios, 0.0)
                number_of_matches = np.append(number_of_matches, 0.0)
                number_of_inliers = np.append(number_of_inliers, 0.0)

        # save data
        with open(outputs / 'qvec_data.json', 'w') as outfile:
            json.dump(qvecs, outfile)
        with open(outputs / 'localization_data.json', 'w') as outfile:
            json.dump(camera_locations_added, outfile)
        with open(outputs / 'transformation_data.json', 'w') as outfile:
            json.dump(transformations, outfile)
        np.savetxt(outputs / 'number_matches.out', number_of_matches)
        np.savetxt(outputs / 'number_inliers.out', number_of_inliers)
        np.savetxt(outputs / 'inlier_ratios.out', inlier_ratios)


    # compute affine transform from raw to corr frame for img with name
    def get_cam_to_cam_transform(self, T_raw, T_corr, name):
        T_raw_cam = np.linalg.inv(T_raw[name])
        T_corr_cam = np.linalg.inv(T_corr[name])

        # vec1 = np.array([[1.0], [0.0], [0.0], [1.0]])
        # vec2 = np.array([[0.0], [1.0], [0.0], [1.0]])
        # vec3 = np.array([[0.0], [0.0], [1.0], [1.0]])
        # vec4 = np.array([[1.0], [1.0], [1.0], [1.0]])

        # M_raw = np.column_stack((np.matmul(T_raw_cam, vec1), np.matmul(T_raw_cam, vec2),
        #                         np.matmul(T_raw_cam, vec3), np.matmul(T_raw_cam, vec4)))
        # M_corr = np.column_stack((np.matmul(T_corr_cam, vec1), np.matmul(T_corr_cam, vec2),
        #                         np.matmul(T_corr_cam, vec3), np.matmul(T_corr_cam, vec4)))

        # T = np.matmul(M_corr, np.linalg.inv(M_raw))
        T = np.matmul(T_corr_cam, np.linalg.inv(T_raw_cam))
        return T

    # load poses and transformations (if transformation_bool=True) before and after alignment
    def load_data(self, raw_path, corrected_path, transformation_bool):
        images_raw = read_images_binary(os.path.join(raw_path, 'images.bin'))
        raw_poses = {}
        for id in images_raw:
            R = images_raw[id].qvec2rotmat()
            pos = np.matmul(-np.linalg.inv(R), images_raw[id].tvec)
            img_name = os.path.basename(images_raw[id].name)
            raw_poses.update({img_name: pos})
        raw_poses = dict(sorted(raw_poses.items()))

        with open(corrected_path + '/data/localization_data.json', "r") as infile:
            data = []
            for line in infile:
                data.append(json.loads(line))
        corr_poses = data[0]

        ground_truth = Evaluation.get_gt_poses(os.path.dirname(self.images_temp_path))

        if transformation_bool == True:
            with open(corrected_path + '/data/transformation_data.json', "r") as infile:
                data = []
                for line in infile:
                    data.append(json.loads(line))
            T_corr = data[0]

            T_raw = {}
            for key in T_corr:
                for id in images_raw:
                    img_name = os.path.basename(images_raw[id].name)
                    if img_name == key:
                        R = images_raw[id].qvec2rotmat()
                        T_mat_raw = [[R[0][0], R[0][1], R[0][2], images_raw[id].tvec[0]],
                                    [R[1][0], R[1][1], R[1][2], images_raw[id].tvec[1]],
                                    [R[2][0], R[2][1], R[2][2], images_raw[id].tvec[2]],
                                    [0.0, 0.0, 0.0, 1.0]]
                        T_raw.update({key: T_mat_raw})
            T = {}
            for name in corr_poses:
                T.update({name: self.get_cam_to_cam_transform(T_raw, T_corr, name)})
            return raw_poses, corr_poses, ground_truth, T
        else:
            return raw_poses, corr_poses, ground_truth


    # compute mean distance of two corresponding cameras in two lists of camera positions
    def get_error_per_cam(self, transformed_points, i, j):
        errors = transformed_points[i] - transformed_points[j]
        total_dist = 0
        for e in errors:
            total_dist += np.linalg.norm(e)
        error_per_camera = total_dist / len(errors)
        return error_per_camera

    # Remove cameras with errors bigger than gps noise and then find inliers and outliers according to transformation
    # matrix. Create plots and save validated cameras as file
    def filter_transformations(self, T, raw_poses, corr_poses, gt_poses):
        errors_raw, errors_corr = [], []
        improved_cams = 0
        corr_poses_filtered, raw_poses_filtered, T_filtered = {}, {}, []
        for img in corr_poses:
            corr = corr_poses[img]
            raw = raw_poses[img]
            gt = gt_poses[img]
            errors_raw.append(np.linalg.norm(np.subtract([gt[0], gt[1], gt[2]], raw)))
            corr_error = np.linalg.norm(np.subtract([gt[0], gt[1], gt[2]], corr))
            if np.linalg.norm(np.subtract(raw, corr)) < 2*self.gps_noise:
                errors_corr.append(corr_error)
                corr_poses_filtered.update({img: corr_poses[img]})
                raw_poses_filtered.update({img: raw_poses[img]})
                T_filtered.append(T[img])
                if errors_raw[-1] > corr_error:
                    improved_cams += 1
            else:
                errors_corr.append(-0.05 * self.gps_noise)

        errors_corr_to_consider = [a for a in errors_corr if a >= 0.0]
        if not errors_corr_to_consider:
            error_text = "Position error \nbefore alignment: \nmean: " + str(round(np.mean(errors_raw), 5)) + \
                        "\nstd dev: " + str(round(np.std(errors_raw), 5)) + "\nAfter alignment: \nno cameras localized"
        else:
            error_text = "Position error \nbefore alignment: \nmean: " + str(round(np.mean(errors_raw), 5)) + \
                        "\nstd dev: " + str(round(np.std(errors_raw), 5)) + "\nAfter alignment: \nmean: " + \
                        str(round(np.mean(errors_corr_to_consider), 5)) + "\nstd dev: " + str(round(np.std(errors_corr_to_consider), 5))
        print(error_text)
        # if self.plotting:
        #     plt.clf()
        #     X = np.arange(len(errors_corr))
        #     a = plt.bar(X + 0.00, errors_raw, color='b', width=0.25)
        #     b = plt.bar(X + 0.25, errors_corr, color='g', width=0.25)
        #     plt.legend((a, b), ('error $C_1^r$', 'error $C_1^c$'),
        #             loc='upper right', fontsize=9)
        #     plt.xticks(X + 0.125, '', fontsize=9)
        #     plt.yticks(fontsize=9)
        #     plt.ylabel('Error in m', fontsize=9)
        #     plt.ylim([-0.025 * self.gps_noise, 1.5 * self.gps_noise])
        #     plt.title('Position errors in camera position')
        #     plt.figtext(0.125, 0.05, 'Number of improved cameras: ' + str(improved_cams) + '/' + str(len(errors_corr)))
        #     plt.tick_params(axis='x', which='both', bottom=False)
        #     plt.savefig(self.output_path + '/data/camera_errors.pdf')
        #     plt.clf()
        #     plt.close('all')

        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     inliers, outliers = self.compute_inlier(T_filtered, raw_poses_filtered, corr_poses_filtered, self.dist_threshold, figure=ax)

        #     raw_inlier, corr_inlier, gt_inlier = np.empty((0, 3), float), np.empty((0, 3), float), np.empty((0, 3), float)
        #     for name in inliers:
        #         raw_inlier = np.append(raw_inlier, [raw_poses[name]], axis=0)
        #         corr_inlier = np.append(corr_inlier, [corr_poses[name]], axis=0)
        #         gt_inlier = np.append(gt_inlier, [[gt_poses[name][0], gt_poses[name][1], gt_poses[name][2]]], axis=0)

        #     raw_outlier, corr_outlier, gt_outlier = np.empty((0, 3), float), np.empty((0, 3), float), np.empty((0, 3), float)
        #     for name in outliers:
        #         raw_outlier = np.append(raw_outlier, [raw_poses[name]], axis=0)
        #         corr_outlier = np.append(corr_outlier, [corr_poses[name]], axis=0)
        #         gt_outlier = np.append(gt_outlier, [[gt_poses[name][0], gt_poses[name][1], gt_poses[name][2]]], axis=0)

        #     x_corr_inlier, y_corr_inlier, z_corr_inlier = zip(*corr_inlier)
        #     x_gt_inlier, y_gt_inlier, z_gt_inlier = zip(*gt_inlier)
        #     x_raw_inlier, y_raw_inlier, z_raw_inlier = zip(*raw_inlier)
        #     x_corr_outlier, y_corr_outlier, z_corr_outlier = zip(*corr_outlier)
        #     x_gt_outlier, y_gt_outlier, z_gt_outlier = zip(*gt_outlier)
        #     x_raw_outlier, y_raw_outlier, z_raw_outlier = zip(*raw_outlier)

        #     error = []
        #     for i in range(len(x_corr_inlier)):
        #         error.append(np.sqrt((x_corr_inlier[i] - x_gt_inlier[i]) ** 2 +
        #                             (y_corr_inlier[i] - y_gt_inlier[i]) ** 2 +
        #                             (z_corr_inlier[i] - z_gt_inlier[i]) ** 2))
        #     error_text = error_text + "\n\nAfter validation: \nmean: " + str(round(np.mean(error), 5)) + \
        #                 "\nstd dev: " + str(round(np.std(error), 5)) + "\nInliers: " + \
        #                 str(len(x_corr_inlier)) + "/" + str(len(x_corr_outlier) + len(x_corr_inlier))

        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     ax.set_xlabel('\n\nX direction', fontsize=9)
        #     ax.set_ylabel('\n\nY direction', fontsize=9)
        #     ax.set_zlabel('\n\nZ direction', fontsize=9)
        #     ax.tick_params(axis='both', which='major', labelsize=7)
        #     ax.tick_params(axis='both', which='minor', labelsize=7)
        #     ax.xaxis.offsetText.set_fontsize(7)
        #     ax.yaxis.offsetText.set_fontsize(7)
        #     ax.set_title('Camera poses before and after temporal alignment')
        #     ax.scatter(x_raw_outlier, y_raw_outlier, z_raw_outlier, c='red', marker="+")
        #     ax.scatter(x_gt_outlier, y_gt_outlier, z_gt_outlier, c='blue', marker="+")
        #     ax.scatter(x_corr_outlier, y_corr_outlier, z_corr_outlier, c='green', marker="+")
        #     r = ax.scatter(x_raw_inlier, y_raw_inlier, z_raw_inlier, c='red')
        #     g = ax.scatter(x_gt_inlier, y_gt_inlier, z_gt_inlier, c='blue')
        #     c = ax.scatter(x_corr_inlier, y_corr_inlier, z_corr_inlier, c='green')
        #     plt.legend((r, c, g), ('Poses $C_1^r$', 'Poses $C_1^c$', 'Poses $C_1^{gt}$'), loc='upper right', fontsize=9)
        #     plt.axis('equal')
        #     plt.figtext(0.02, 0.35, error_text, fontsize=9)
        #     plt.savefig(self.output_path + '/data/camera_poses.pdf')
        #     plt.clf()
        #     plt.close('all')

        #     print("plots created")

        # else:
        inliers, outliers = self.compute_inlier(T_filtered, raw_poses_filtered, corr_poses_filtered, self.dist_threshold)

        with open(self.output_path + '/data/inlier_GPS.txt', 'w') as f:
            for img_name in inliers:
                coords = corr_poses[img_name]
                img_name = os.path.join(self.images_temp_relative_path, img_name)
                f.write(img_name + ' ' + str(coords[0]) + ' ' + str(coords[1]) + ' ' + str(coords[2]) + '\n')
        print("inlier_GPS.txt created in .../data/")

        return inliers, outliers
    
    def cluster_based_outlier_detection(self, quaternions, min_samples, eps=0.1):
        # Use DBSCAN for clustering-based outlier detection
        X = np.array(quaternions)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_
        inliers = [i for i, label in enumerate(labels) if label != -1]
        return inliers

    def filter_transformations_new(self, T, raw_poses, corr_poses, gt_poses):
        with open(self.output_path + '/data/qvec_data.json', "r") as infile:
            data = []
            for line in infile:
                data.append(json.loads(line))
        qvecs = data[0]

        quaternions = np.empty((0, 4), float)
        for qvec in qvecs.values():
            quaternions = np.append(quaternions, [qvec], axis=0)

        min_samples = int(len(qvecs) * 0.2)  # Adjust the minimum number of samples in a cluster as needed
        cluster_inliers = self.cluster_based_outlier_detection(quaternions, min_samples)
        inliers = [list(qvecs.keys())[i] for i in cluster_inliers]
        print(f"Find {len(inliers)}/{len(qvecs)} inliers")

        with open(self.output_path + '/data/inlier_GPS.txt', 'w') as f:
            for img_name in inliers:
                coords = corr_poses[img_name]
                img_name = os.path.join(self.images_temp_relative_path, img_name)
                f.write(img_name + ' ' + str(coords[0]) + ' ' + str(coords[1]) + ' ' + str(coords[2]) + '\n')
        print("inlier_GPS.txt created in .../data/")


    # Transform raw_poses with different T first. Then compute distances to other transformed points and
    # get inliers and outliers
    def compute_inlier(self, T, raw_poses, corr_poses, distance_threshold, figure=None):
        print(f'computing inliers...')
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
                    distance = self.get_error_per_cam(transformed_points, idx_i, idx_j)
                    distance_mat[idx_i][idx_j] = distance
                    distance_mat[idx_j][idx_i] = distance
        dist = []
        for i in range(len(distance_mat)):
            dist.append(sum(distance_mat[i]))
        # min_names = [list(raw_poses.keys())[i] for i in min_idx]
        # print('images with least error', min_names)
        num_rows_to_consider = len(distance_mat) // 2
        num_inliers = len(distance_mat) // 10
        min_row_idx = sorted(range(len(dist)), key=lambda sub: dist[sub])[:num_rows_to_consider]
        votes = [0] * len(distance_mat)
        for row_id in min_row_idx:
            dist_row = distance_mat[row_id]
            min_idx = sorted(range(len(dist_row)), key=lambda sub: dist_row[sub])[:num_inliers]
            for i in min_idx:
                votes[i] += 1
        inliers = sorted(range(len(votes)), key=lambda sub: votes[sub])[:num_inliers]
        outliers = list(set(range(len(distance_mat)))-set(inliers))
        # center_idx = dist.index(min(dist))
        # outliers = []
        # done = False
        # num_inliers_threshold = 3
        # while done == False:
        #     inliers = [center_idx]
        #     outliers = []
        #     for i in range(len(distance_mat)):
        #         if distance_mat[center_idx][i]:
        #             if distance_mat[center_idx][i] < distance_threshold:
        #                 inliers.append(i)
        #             else:
        #                 outliers.append(i)
        #     print(f'distance_threshold: {distance_threshold}, num_inliers: {len(inliers)}')
        #     if len(inliers) > num_inliers_threshold:
        #         done = True
        #     else:
        #         if distance_threshold < 1.0:
        #             distance_threshold += 0.05
        #         else:
        #             done=True

        if figure is not None:
            for i in inliers:
                x_trans_list, y_trans_list, z_trans_list = zip(*transformed_points[i])
                inl = figure.scatter(x_trans_list, y_trans_list, z_trans_list, color=[0.0, 1.0, 0.0], marker="+")
            for o in outliers:
                x_trans_list, y_trans_list, z_trans_list = zip(*transformed_points[o])
                outl = figure.scatter(x_trans_list, y_trans_list, z_trans_list, color='black', marker="+", s=5)
            plt.legend((inl, outl), ('inliers', 'outliers'), loc='upper left', fontsize=5)

            figure.set_xlabel('\n\nX direction', fontsize=9)
            figure.set_ylabel('\n\nY direction', fontsize=9)
            figure.set_zlabel('\n\nZ direction', fontsize=9)
            figure.tick_params(axis='both', which='major', labelsize=7)
            figure.tick_params(axis='both', which='minor', labelsize=7)
            figure.xaxis.offsetText.set_fontsize(7)
            figure.yaxis.offsetText.set_fontsize(7)
            figure.set_title('Inliers and Outliers')
            plt.clf()
            plt.close('all')

        names_inlier = []
        for i in inliers:
            names_inlier.append(list(raw_poses.keys())[i])

        names_outlier = []
        for o in outliers:
            names_outlier.append(list(raw_poses.keys())[o])

        return names_inlier, names_outlier


    # use colmaps model aligner to find similarity transform to align validated cameras
    def correct_model(self):
        Reconstruction.align_with_gps(output_dir=self.output_path,
                                    model_input=os.path.join(self.reconstruction_temp_path), 
                                    model_output=os.path.join(self.output_path, 'sparse/corrected'), 
                                    reference=os.path.join(self.output_path, 'data/inlier_GPS.txt'), 
                                    logname='correction_output')

    # extract features and localize cameras of temp model in ref model. Then validate the localization and
    # align model with validated cameras
    def run(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # self.localize_cameras()

        raw_poses, corr_poses, gt_poses, T = self.load_data(self.reconstruction_temp_path, self.output_path, True)
        inlier_list, outlier_list = self.filter_transformations(T, raw_poses, corr_poses, gt_poses)
        # self.filter_transformations_new(T, raw_poses, corr_poses, gt_poses)

        try:
            self.correct_model()
        except:
            self.is_successful = False

if __name__ == "__main__":
    start_time = time.time()

    basedir = '/path/to/experiment'
    reconstruction_ref_path = '/path/to/reconstruction_ref'
    reconstruction_temp_path = '/path/to/reconstruction_temp'
    images_ref_path = '/path/to/images_ref'
    images_temp_path = '/path/to/images_temp'
    output_path = basedir + '/Superpoint_custom/20180402'
    
    localization = CameraLocalization(output_path, images_ref_path, images_temp_path, 
                                      reconstruction_ref_path, reconstruction_temp_path,
                                      extractor='superpoint_max', matcher='superglue',
                                      plotting=False, gps_noise=5.0, dist_threshold=0.20)
    localization.main()

    end_time = time.time()
    run_time = end_time - start_time
    print("Runtime: ", run_time)