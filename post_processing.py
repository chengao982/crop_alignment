import numpy as np
import time
import os


# run the whole pipeline
def main(path):
    methods = ['SIFT', 'Superpoint', 'Superpoint_custom']
    for method in methods:
        workspace_path = path + method
        save_text = 'Statistics of temporal matching'
        subdirectories = [file for file in os.listdir(workspace_path) if
                          os.path.isdir(os.path.join(workspace_path, file))]
        for folder in subdirectories:
            data_path = workspace_path + '/' + folder + '/data'
            number_inliers = np.loadtxt(data_path + '/number_inliers.out')
            number_matches = np.loadtxt(data_path + '/number_matches.out')
            inlier_ratios = np.loadtxt(data_path + '/inlier_ratios.out')
            text = '\n\n***** SIFT ' + folder + ' *****\nnumber_matches:\nmean: ' + str(round(np.mean(number_matches), 2)) + \
                   '\nstd dev: ' + str(round(np.std(number_matches), 2)) + '\nmedian: ' + str(round(np.median(number_matches), 2)) + \
                   '\nmax: ' + str(np.max(number_matches)) + '\nmin: ' + str(np.min(number_matches)) + \
                   '\nnumber_inliers:\nmean: ' + str(round(np.mean(number_inliers), 2)) + \
                   '\nstd dev: ' + str(round(np.std(number_inliers), 2)) + '\nmedian: ' + str(round(np.median(number_inliers), 2)) + \
                   '\nmax: ' + str(np.max(number_inliers)) + '\nmin: ' + str(np.min(number_inliers)) + \
                   '\ninlier_ratios:\nmean: ' + str(round(np.mean(inlier_ratios), 2)) + \
                   '\nstd dev: ' + str(round(np.std(inlier_ratios), 2)) + '\nmedian: ' + str(round(np.median(inlier_ratios), 2)) + \
                   '\nmax: ' + str(np.max(inlier_ratios)) + '\nmin: ' + str(np.min(inlier_ratios))
            save_text += text
        with open(workspace_path + '/statistics.txt', "w") as text_file:
            text_file.write(save_text)


if __name__ == "__main__":
    start_time = time.time()
    #TODO: adapt this path
    project_path = '/path/to/experiment/'

    main(project_path)

    end_time = time.time()
    run_time = end_time - start_time
    print("Runtime: ", run_time)
