import math
import os
import subprocess
import read_write_model
import matplotlib.pyplot as plt
import numpy as np
from xml.dom import minidom
from scipy.optimize import fsolve
import time
import open3d as o3d
import json

# returns the markers position in img coordinates and marker position in GPS coords
def read_marker_img_pos(workspace_dir, data_dir, translation_coords):
    file = minidom.parse(data_dir)
    cameras = file.getElementsByTagName('camera')
    markers = file.getElementsByTagName('marker')
    dict_m = {}

    img_path = workspace_dir + '/images'
    used_images = sorted([f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))])
    for img in used_images:
        for elem in cameras:
            if elem.attributes['label'].value == img:
                img_id = elem.attributes['id'].value
                for elem in markers:
                    try:
                        marker = elem.attributes['marker_id'].value
                        locations = elem.getElementsByTagName('location')
                        for i in locations:
                            if i.attributes['camera_id'].value == img_id:
                                data = [img, [float(i.attributes['x'].value), float(i.attributes['y'].value)]]
                                if marker in dict_m:
                                    current = dict_m[marker]
                                    current.append(data)
                                    dict_m.update({marker: current})
                                else:
                                    dict_m.update({marker: [data]})
                    except:
                        pass
    # remove markers that are not at least seen in two images and get GPS pose of remaining markers
    ground_truth = {}
    to_delete = []
    for i in dict_m:
        if len(dict_m[i]) < 2:
            to_delete.append(i)
        else:
            for elem in markers:
                try:
                    marker = elem.attributes['id'].value
                    if i == marker:
                        references = elem.getElementsByTagName('reference')
                        for ref in references:
                            ground_truth.update({marker: [float(ref.attributes['x'].value) + translation_coords[0],
                                                          float(ref.attributes['y'].value) + translation_coords[1],
                                                          float(ref.attributes['z'].value) + translation_coords[2]]})
                except:
                    pass
    for i in to_delete:
        del dict_m[i]

    return dict(sorted(dict_m.items())), dict(sorted(ground_truth.items()))

# compute intersection point of rays of form g = a +lambda*r with least squares
def get_intersection_ls(a, r):
    s_mat = np.zeros((3, 3))
    c_mat = np.zeros((3, 1))
    for x in range(len(r)):
        # normalize r vectors and then compute s = sum(normalized_i*normalized_i.T - eye) and
        # c = sum((normalized_i*normalized_i.T - eye) * origin_i)
        normalized = np.array([[r[x][0]], [r[x][1]], [r[x][2]]]) / math.sqrt(r[x][0]**2 + r[x][1]**2 + r[x][2]**2)
        s = np.dot(normalized, np.transpose(normalized)) - np.identity(3)
        origin = np.array([[a[x][0]], [a[x][1]], [a[x][2]]])
        c = np.matmul(s, origin)
        s_mat = np.add(s_mat, s)
        c_mat = np.add(c_mat, c)

    # solve s_mat * point = c_mat
    point = np.matmul(np.linalg.pinv(s_mat), c_mat)

    # compute max error
    errors = []
    for x in range(len(r)):
        normalized = np.array([r[x][0], r[x][1], r[x][2]]) / math.sqrt(r[x][0] ** 2 + r[x][1] ** 2 + r[x][2] ** 2)
        origin = np.array([[a[x][0]], [a[x][1]], [a[x][2]]])
        vec = np.reshape(np.subtract(origin, point), 3)
        d = np.cross(vec, normalized)
        dist = np.linalg.norm(d)
        errors.append(dist)
    # print("Max error of ray to intersection point: " + str(round(max(errors), 5)))
    if max(errors)>0.5 and len(r)>2:
        print("Recomputing ... error over 0.5m")
        max_index = np.argmax(errors)
        del r[max_index]
        del a[max_index]
        point = get_intersection_ls(a, r)

    return point

# compute the GCP position in gps frame from the dataset
def get_marker_gps_position(markers, images, dir, from_reconstruction, model):
    markers_from_data = {}
    if from_reconstruction == False:
        camera_poses = read_camera_poses_from_data(dir)
    for id in markers:
        a_list = []
        r_list = []
        for observation in markers[id]:
            img_name = observation[0]
            for idx in images.keys():
                img = images.get(idx)
                if img_name == img.name:
                    K, distortion = get_calibration_matrix(dir, img.camera_id, from_reconstruction, model)
                    img_coords = np.array([[observation[1][0]], [observation[1][1]], [1.0]])
                    tdist = np.matmul(np.linalg.inv(K), img_coords)
                    def radial_dist_equations(p):
                        x, y = p
                        return (x + distortion * (x ** 3 + y ** 2) - tdist[0][0],
                                y + distortion * (x ** 2 + y ** 3) - tdist[1][0])
                    x, y = fsolve(radial_dist_equations, (1, 1))

                    if from_reconstruction == True:
                        R = img.qvec2rotmat()
                        Tr = img.tvec
                        a = np.matmul(-np.linalg.inv(R), np.array([[Tr[0]], [Tr[1]], [Tr[2]]]))
                    else:
                        a = np.array(
                            [[camera_poses[img_name][0]], [camera_poses[img_name][1]], [camera_poses[img_name][2]]])
                        R = camera_poses[img_name][3]

                    tvec = np.matmul(np.linalg.inv(R), np.array([[x], [y], [1.0]]))

                    r_list.append([tvec[0][0], tvec[1][0], tvec[2][0]])
                    a_list.append([a[0][0], a[1][0], a[2][0]])

        p = get_intersection_ls(a_list, r_list)
        markers_from_data.update({id: [p[0][0], p[1][0], p[2][0]]})
    return dict(sorted(markers_from_data.items()))

# get camera poses from dataset
def read_camera_poses_from_data(dir):
    file_name = dir + '/data/camera_position_JPEG.txt'
    translation_coords = np.loadtxt(dir + '/data/translation_vector.txt')
    poses = {}
    with open(file_name) as f:
        lines = f.readlines()
    for i in range(2, len(lines)):
        data = lines[i][:-1].split()
        rotMat = np.array([[float(data[7]), float(data[8]), float(data[9])], [-float(data[10]), -float(data[11]), -float(data[12])], [-float(data[13]), -float(data[14]), -float(data[15])]])
        coords = [float(data[1]) + translation_coords[0], float(data[2]) + translation_coords[1], float(data[3]) + translation_coords[2], rotMat]
        poses.update({data[0]: coords})
    poses = dict(sorted(poses.items()))
    return poses

# read calibration matrix from reconstruction or from dataset otherwise
def get_calibration_matrix(dir, cam_id, from_reconstruction, model):
    if from_reconstruction == True:
        cameras = read_write_model.read_cameras_binary(dir + '/sparse/' + model + '/cameras.bin')
        # f, cx, cy, k
        parameters = cameras[cam_id].params
        K = np.array([[parameters[0], 0.0, parameters[1]], [0.0, parameters[0], parameters[2]], [0.0, 0.0, 1.0]])
        k1 = parameters[3]

    else:
        file = minidom.parse(dir + '/data/markers_placed_JPEG.xml')
        sensor = file.getElementsByTagName('sensor')
        resolution = sensor[0].getElementsByTagName('resolution')[0]
        width = float(resolution.attributes['width'].value)
        height = float(resolution.attributes['height'].value)

        f = float(sensor[0].getElementsByTagName('f')[0].firstChild.nodeValue)
        cx = width / 2 + float(sensor[0].getElementsByTagName('cx')[0].firstChild.nodeValue)
        cy = height / 2 + float(sensor[0].getElementsByTagName('cy')[0].firstChild.nodeValue)
        k1 = float(sensor[0].getElementsByTagName('k1')[0].firstChild.nodeValue)
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return K, k1

# error computation for plotting
def plot_preprocessing(markers_data, markers_ground_truth, title, path):
    data_list = np.empty((0, 3), float)
    ground_truth_list = np.empty((0, 3), float)
    error2D_list = []
    error3D_list = []
    names = []
    error_text = ""
    for id in markers_data:
        data_list = np.append(data_list, [markers_data[id]], axis=0)
        ground_truth_list = np.append(ground_truth_list, [markers_ground_truth[id]], axis=0)
        error2D = np.sqrt(np.power(markers_data[id][0] - markers_ground_truth[id][0], 2) +
                          np.power(markers_data[id][1] - markers_ground_truth[id][1], 2))
        error3D = np.sqrt(np.power(markers_data[id][0] - markers_ground_truth[id][0], 2) +
                          np.power(markers_data[id][1] - markers_ground_truth[id][1], 2) +
                          np.power(markers_data[id][2] - markers_ground_truth[id][2], 2))
        error2D_list.append(error2D)
        error3D_list.append(error3D)
        error_text += "Marker " + str(id) + ":\nError in 3D: " + str(round(error3D, 5)) + "\nError in 2D: " + \
                      str(round(error2D, 5)) + "\n\n"
        names.append("M" + str(id))
    print(error_text)
    print("Error mean 3D: " + str(round(np.mean(error3D_list), 5)) +
          "\nError std dev 3D: " + str(round(np.std(error3D_list), 5)) +
          "\n\nError mean 2D: " + str(round(np.mean(error2D_list), 5)) +
          "\nError std dev 2D: " + str(round(np.std(error2D_list), 5)))

    with open(path + '/' + title + '_errors.txt', 'w') as f:
        f.write(error_text)

    names.append('Mean')
    error2D_list.append(np.mean(error2D_list))
    error3D_list.append(np.mean(error3D_list))
    X = np.arange(len(names))
    a = plt.bar(X + 0.00, error2D_list, color='b', width=0.25)
    b = plt.bar(X + 0.25, error3D_list, color='g', width=0.25)
    plt.legend((a, b), ('error in 2D', 'error in 3D'),
               loc='upper right', fontsize=9)
    plt.xticks(X + 0.125, names, fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylabel('Error in m', fontsize=9)
    plt.title('Position errors in ground control points')
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.savefig(path + '/' + title + '_errors.pdf')
    plt.clf()

    return data_list, ground_truth_list, error_text, error3D_list, error2D_list, names

# create plots
def plot_GCP(markers_data, markers_ground_truth, title, dir):
    p = os.path.join(dir, 'output/plots')
    if not os.path.exists(p):
        os.makedirs(p)

    data_list, ground_truth_list, error_text, e_list_3D, e_list_2D, names = plot_preprocessing(markers_data, markers_ground_truth, title, p)

    x_data, y_data, z_data = zip(*data_list)
    x_ground_truth, y_ground_truth, z_ground_truth = zip(*ground_truth_list)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('\n\nX direction', fontsize=9)
    ax.set_ylabel('\n\nY direction', fontsize=9)
    ax.set_zlabel('\n\nZ direction', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    ax.xaxis.offsetText.set_fontsize(7)
    ax.yaxis.offsetText.set_fontsize(7)
    ax.set_title('Position of ground control points')

    d = ax.scatter(x_data, y_data, z_data, c='red')
    g = ax.scatter(x_ground_truth, y_ground_truth, z_ground_truth, c='blue')
    plt.legend((d, g), ('GCP poses from reconstruction', 'Ground truth from data'),
               loc='upper left', fontsize=9)
    plt.savefig(p + '/' + title + '_scaled_axis.pdf')
    plt.axis('equal')

    plt.savefig(p + '/' + title + '.pdf')
    plt.clf()

    p = os.path.join(dir, 'output/details')
    if not os.path.exists(p):
        os.makedirs(p)

    with open(p + '/' + title + '_data.json', 'w') as outfile:
        json.dump('# dicts with GCP positions from reconstruction and GCP positions from dataset', outfile)
        outfile.write('\n')
        json.dump(markers_data, outfile)
        outfile.write('\n')
        json.dump(markers_ground_truth, outfile)

# save reconstruction in txt format and create pointcloud
def convert_to_txt(dir):
    p = os.path.join(dir, 'output/details/aligned')
    if not os.path.exists(p):
        os.makedirs(p)

    logfile_name = os.path.join(dir, 'output/details/colmap_output.txt')
    logfile = open(logfile_name, 'w')

    feature_extractor_args = [
        'colmap', 'model_converter',
        '--input_path', os.path.join(dir, 'sparse/aligned'),
        '--output_path', os.path.join(dir, 'output/details/aligned'),
        '--output_type', 'TXT',
    ]
    converter_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
    logfile.write(converter_output)

    file_path = dir + '/output/details/aligned/points3D.txt'
    with open(file_path) as f:
        lines = f.readlines()
    f.close()
    del lines[0:3]

    raw_file_path = dir + '/output/details/features_raw.txt'
    with open(raw_file_path, 'w') as f:
        for line in lines:
            data = line.split()
            mystring = data[1] + ' ' + data[2] + ' ' + data[3] + '\n'
            f.write(mystring)
    print('Model converted to txt file')

    file_data = np.loadtxt(raw_file_path, dtype=float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(file_data)
    pointcloud_path = dir + '/output/details/features_raw.pcd'
    o3d.io.write_point_cloud(pointcloud_path, pcd)
    print('Pointcloud created')

def main(workspace_path, model):
    images = read_write_model.read_images_binary(workspace_path + '/sparse/aligned/images.bin')
    translation_coords = np.loadtxt(workspace_path + '/data/translation_vector.txt')

    markers, markers_gps_pos = read_marker_img_pos(workspace_path, workspace_path + '/data/markers_placed_JPEG.xml', translation_coords)
    markers_reconstruction = get_marker_gps_position(markers, images, workspace_path, True, model)
    markers_data = get_marker_gps_position(markers, images, workspace_path, False, model)

    plot_GCP(markers_reconstruction, markers_data, 'GCP_positions', workspace_path)
    convert_to_txt(workspace_path)

if __name__ == "__main__":
    start_time = time.time()
    # TODO: adapt this path
    workspace_path = '/path/to/reconstruction'
    model = 'aligned'
    main(workspace_path, model)
    end_time = time.time()
    run_time = end_time - start_time
    print("Runtime: ", run_time)
