import os
from generate_image_poses import ImagePoseGenerator
from reconstructor import Reconstruction
from localizer import CameraLocalization
from evaluator import Evaluation
from query_ref_pair import pairs
import pandas as pd
import numpy as np
import time
import shutil
import warnings

class ReconstructionPipeline:
    def __init__(self, 
                 data_path, 
                 output_path, 
                 source_images_path, 
                 initial_models_path,
                 image_poses_file_name, 
                 experiment_name, 
                 extractor_matchers,
                 pairs_dict,
                 gps_error=5.0, 
                 distance_threshold=0.20, 
                 use_previous_as_ref=False
                 ):
        
        self.data_path = data_path
        self.output_path = os.path.join(output_path, experiment_name)
        self.source_images_path = source_images_path
        self.image_poses_file_name = image_poses_file_name
        self.initial_models_path = initial_models_path
        self.extractor_matchers = extractor_matchers
        self.pairs_dict = pairs_dict
        self.plot = True
        self.gps_error = gps_error
        self.distance_threshold = distance_threshold
        self.use_previous_as_ref = use_previous_as_ref
        subfolders = next(os.walk(self.data_path))[1]
        self.subfolders = sorted(subfolders)

        self.query_list = list(pairs_dict.keys())
        self.num_ref_bins = len(pairs_dict[self.query_list[0]])

        self.output_df_dict = self.create_output_df(row_names='alg', col_names='ref')


    def create_output_df(self, row_names, col_names):
        # row_names = [extractor if extractor else matcher for (extractor, matcher) in self.extractor_matchers]
        # col_names = list(range(self.num_ref_bins))

        names_dict = {
            'alg': [extractor if extractor else matcher for (extractor, matcher) in self.extractor_matchers],
            'ref': list(range(self.num_ref_bins)),
            'query': self.query_list
        }

        template_df = pd.DataFrame(index=names_dict[row_names], columns=names_dict[col_names])

        df_names = ['dt_mean', 'dt_std', 'dr_mean', 'dr_std', 'error3D_mean', 'error3D_std', 'error2D_mean', 'error2D_std']
        df_dict = {}

        for name in df_names:
            template_copy = template_df.__deepcopy__()
            df_dict[name] = template_copy

        return df_dict
        
    def save_output_df(self, output_df_dict, output_df_name):
        # Create a Pandas Excel writer object
        with pd.ExcelWriter(output_df_name, engine='xlsxwriter') as writer:
            # Loop through the dictionary and write each DataFrame to a separate sheet
            for sheet_name, df in output_df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)


    def generate_poses(self, polygon_corners, minimum_distance=0):
        '''Generate image_poses.txt with interested images that will be used for reconstruction and localization 

        Args: 
        polygon_corners: list of corner xy coordinates, clock-wise or counter clock-wise.
                         specifies the boarder of the interested images. 
                         only images inside it will be included in image_poses.txt
        minimum_distance: the minimum distance between interested images. 
                          will discard image whose distance to its neareat neighbor is smaller than this parameter
        '''
        for subfolder in self.subfolders:
            print('--------------------Pose File Generation--------------------')
            print(f"Generating image_poses.txt for subfolder {subfolder}\n")

            generator = ImagePoseGenerator(os.path.join(self.data_path, subfolder), self.image_poses_file_name)
            generator.process_camera_poses(polygon_corners, distance_threshold=minimum_distance)

        print('====================Pose File Generation Done====================\n')

    # def build_inital_models(self):
    #     for idx, subfolder in enumerate(self.subfolders):
    #         start_time = time.time()
    #         print('--------------------Intial Reconstruction--------------------')
    #         print(f"Running intial reconstruction for subfolder {subfolder}...\n")
            
    #         output_path = os.path.join(self.initial_models_output_path, subfolder)
    #         data_path = os.path.join(self.data_path, subfolder)
    #         source_images_path = os.path.join(self.source_images_path, subfolder, 'RAW/JPEG')

    #         if not os.path.isfile(os.path.join(output_path, 'rec_done.txt')):

    #             if idx == 0:
    #                 print("Running reconstructor ground truth model ...\n")
    #                 reconstructor = Reconstruction(data_path=data_path, 
    #                                             output_path=output_path, 
    #                                             source_images_path=source_images_path,
    #                                             image_poses_file_name=self.image_poses_file_name,
    #                                             error=0.0)
    #                 reconstructor.run()

    #             else:
    #                 print(f"Running reconstructor temporal model {idx} ...\n")
    #                 reconstructor = Reconstruction(data_path=data_path, 
    #                                             output_path=output_path, 
    #                                             source_images_path=source_images_path,
    #                                             image_poses_file_name=self.image_poses_file_name,
    #                                             error=self.gps_error)
    #                 reconstructor.run()

    #             # done flag
    #             with open(os.path.join(output_path, 'rec_done.txt'), 'w') as f:
    #                 f.write('done')

    #         else:
    #             print(f'Initial model of {subfolder} already exists\n')

    #         end_time = time.time()
    #         run_time = end_time - start_time
    #         print(f"Initial Reconstruction Runtime for {subfolder}: {run_time}\n")

    #     print('====================Intial Reconstruction Done====================\n')

    # def evalate_reconstruction(self, translation_error_thres, rotation_error_thres, ground_dist_thres):
    #     for subfolder in self.subfolders:
    #         start_time = time.time()
    #         print('-----------------Reconstruction Evaluation-----------------')
    #         print(f"Running evaulation for subfolder {subfolder}...")

    #         output_path = os.path.join(self.initial_models_output_path, subfolder)
    #         data_gt_path = os.path.join(self.data_path, subfolder)
    #         reconstruction_path = os.path.join(output_path, 'sparse/aligned')

    #         if not os.path.isfile(os.path.join(output_path, 'eval_done.txt')):
    #             evaluator = Evaluation(data_gt_path=data_gt_path,
    #                                 output_path=output_path,
    #                                 reconstruction_path=reconstruction_path,
    #                                 image_poses_file_name=self.image_poses_file_name,
    #                                 translation_error_thres=translation_error_thres,
    #                                 rotation_error_thres=rotation_error_thres,
    #                                 ground_dist_thres=ground_dist_thres)
    #             evaluator.run()

    #             # done flag
    #             with open(os.path.join(output_path, 'eval_done.txt'), 'w') as f:
    #                 f.write('done')
    #         else:
    #             print(f'Evaluation for {subfolder} has already been done\n')

    #         end_time = time.time()
    #         run_time = end_time - start_time
    #         print(f"Reconstruction Evaulation Runtime for {subfolder}: {run_time}\n")

    #     print('====================Reconstruction Evaluation Done====================\n')

    def _localize_cameras(self, extractor, matcher, ref_bin_idx, query_folder):
        identifier = extractor if extractor else matcher

        start_time = time.time()
        
        ref_folder = self.pairs_dict[query_folder][ref_bin_idx]

        output_path = os.path.join(self.output_path, identifier, str(ref_bin_idx), query_folder)
        data_ref_path = os.path.join(self.data_path, ref_folder)
        data_temp_path = os.path.join(self.data_path, query_folder)
        reconstruction_temp_path = os.path.join(self.initial_models_path, query_folder, 'sparse/noisy')
        reconstruction_ref_path = os.path.join(self.initial_models_path, ref_folder, 'sparse/gt')

        if (not os.path.isfile(os.path.join(output_path, 'loc_failed.txt'))) \
            and (not os.path.isfile(os.path.join(output_path, 'loc_done.txt'))):

            localizer = CameraLocalization(output_path=output_path,
                                        images_ref_path=os.path.join(data_ref_path, 'images4reconstruction'),
                                        images_temp_path=os.path.join(data_temp_path, 'images4reconstruction'),
                                        reconstruction_ref_path=reconstruction_ref_path,
                                        reconstruction_temp_path=reconstruction_temp_path,
                                        image_poses_file_name=self.image_poses_file_name,
                                        extractor=extractor,
                                        matcher=matcher,
                                        gps_noise=self.gps_error,
                                        dist_threshold=self.distance_threshold,
                                        plotting=True)
            localizer.run()

            if localizer.is_successful is False:
                with open(os.path.join(output_path, 'loc_failed.txt'), 'w') as f:
                    f.write('failed')
                print('Localization failed')
            else:
                with open(os.path.join(output_path, 'loc_done.txt'), 'w') as f:
                    f.write('done')

        else:
            print('Localization has already been done\n')

        end_time = time.time()
        run_time = end_time - start_time
        print(f"Localization Runtime: {run_time}\n")

        return not os.path.isfile(os.path.join(output_path, 'loc_failed.txt'))

    def _evalate_localization(self, extractor, matcher, ref_bin_idx, query_folder,
                              translation_error_thres, rotation_error_thres, ground_dist_thres):
        
        identifier = extractor if extractor else matcher

        start_time = time.time()

        output_path = os.path.join(self.output_path, identifier, str(ref_bin_idx), query_folder)
        data_gt_path = os.path.join(self.data_path, query_folder)
        reconstruction_path = os.path.join(output_path, 'sparse/corrected')

        print('-----------------corrected_model-----------------')
        evaluator = Evaluation(data_gt_path=data_gt_path,
                            output_path=output_path,
                            reconstruction_path=reconstruction_path,
                            image_poses_file_name=self.image_poses_file_name,
                            translation_error_thres=translation_error_thres,
                            rotation_error_thres=rotation_error_thres,
                            ground_dist_thres=ground_dist_thres)
        dt_mean, dr_mean, error3D_mean, error2D_mean = evaluator.run()

        print('----------------localized_poses------------------')
        evaluator.run_localized()

        end_time = time.time()
        run_time = end_time - start_time
        print(f"Evaulation Runtime: {run_time}\n")

        return dt_mean, dr_mean, error3D_mean, error2D_mean

    def localize_cameras(self, translation_error_thres, rotation_error_thres, ground_dist_thres):
        for extractor_matcher in self.extractor_matchers:
            extractor, matcher = extractor_matcher
            identifier = extractor if extractor else matcher

            alg_output_df_dict = self.create_output_df(row_names='query', col_names='ref')
            print(f'==========Start localization with {extractor} / {matcher}==========\n')

            for ref_bin_idx in range(self.num_ref_bins):
                print(f'==========Start localization for reference bin #{ref_bin_idx}==========\n')

                dt = []
                dr = []
                error3D = []
                error2D = []

                all_queries_sussessful = True

                for query_folder in self.query_list:
                    ref_folder = self.pairs_dict[query_folder][ref_bin_idx]
                    if ref_folder is not None:
                        print(f'-----------------{identifier} Localization, reference bin #{ref_bin_idx}-----------------')
                        print(f"Query-Ref pair: {query_folder} - {ref_folder}")
                        localization_successful = self._localize_cameras(extractor, matcher, ref_bin_idx, query_folder)

                        if localization_successful:
                            print(f'-----------------{identifier} Evaluation, reference bin #{ref_bin_idx}-----------------')
                            print(f"Query-Ref pair: {query_folder} - {ref_folder}")
                            eval_output = self._evalate_localization(
                                            extractor, matcher, ref_bin_idx, query_folder,
                                            translation_error_thres, rotation_error_thres, ground_dist_thres)
                            dt.append(eval_output['dt_mean'])
                            dr.append(eval_output['dr_mean'])
                            error3D.append(eval_output['error3D_mean'])
                            error2D.append(eval_output['error2D_mean'])
                            alg_output_df_dict['dt_mean'][query_folder, ref_bin_idx] = eval_output['dt_mean']
                            alg_output_df_dict['dt_std'][query_folder, ref_bin_idx] = eval_output['dt_std']
                            alg_output_df_dict['dr_mean'][query_folder, ref_bin_idx] = eval_output['dr_mean']
                            alg_output_df_dict['dr_std'][query_folder, ref_bin_idx] = eval_output['dr_std']
                            alg_output_df_dict['error3D_mean'][query_folder, ref_bin_idx] = eval_output['error3D_mean']
                            alg_output_df_dict['error3D_std'][query_folder, ref_bin_idx] = eval_output['error3D_std']
                            alg_output_df_dict['error2D_mean'][query_folder, ref_bin_idx] = eval_output['error2D_mean']
                            alg_output_df_dict['error2D_std'][query_folder, ref_bin_idx] = eval_output['error2D_std']
                        else:
                            all_queries_sussessful = False
                            for name in alg_output_df_dict.keys():
                                alg_output_df_dict[name][query_folder, ref_bin_idx] = pd.NA
                    
                    if all_queries_sussessful:
                        self.output_df_dict['dt_mean'][identifier, ref_bin_idx] = np.mean(dt)
                        self.output_df_dict['dt_std'][identifier, ref_bin_idx] = np.std(dt)
                        self.output_df_dict['dr_mean'][identifier, ref_bin_idx] = np.mean(dr)
                        self.output_df_dict['dr_std'][identifier, ref_bin_idx] = np.std(dr)
                        self.output_df_dict['error3D_mean'][identifier, ref_bin_idx] = np.mean(error3D)
                        self.output_df_dict['error3D_std'][identifier, ref_bin_idx] = np.std(error3D)
                        self.output_df_dict['error2D_mean'][identifier, ref_bin_idx] = np.mean(error2D)
                        self.output_df_dict['error2D_std'][identifier, ref_bin_idx] = np.std(error2D)
                    else:
                        for name in self.output_df_dict.keys():
                            self.output_df_dict[name][identifier, ref_bin_idx] = pd.NA

                self.save_output_df(alg_output_df_dict, os.path.join(self.output_path, identifier, 'alg_eval.xlsx'))
                print(f'==========Finished localization for reference bin #{ref_bin_idx}==========\n')

        self.save_output_df(self.output_df_dict, os.path.join(self.output_path, 'output_df.xlsx'))

    # def evalate_localization(self, translation_error_thres, rotation_error_thres, ground_dist_thres):
    #     for extractor_matcher in self.extractor_matchers:
    #         extractor, matcher = extractor_matcher
    #         self._evalate_localization(extractor, matcher, translation_error_thres, rotation_error_thres, ground_dist_thres)
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # data_path = '/Volumes/Plextor/20190313_20190705_int16'
    # output_path = '/Volumes/Plextor/output'
    # source_images_path = '/Volumes/Plextor/crops'
    # data_path = '/home/gao/dataset_loftr/crop/real_first_month'
    # output_path = '/home/gao/crop_alignment/output'
    # source_images_path = '/mnt/buzz_newhd/home/v4rl/pheno-datasets'

    data_path = '/home/gao/dataset_loftr/crop/crop_2019'
    output_path = '/home/gao/crop_alignment/output'
    source_images_path = '/mnt/usb-ROG_ESD-S1C_N5D0AP040191-0:0'
    initial_models_path = os.path.join(output_path, 'crop_2019', 'initial_models')

    experiment_name = 'exp_v1'

    image_poses_file_name = 'image_poses_tight.txt'

    extractor_matchers = [
                        ['sift', 'NN-ratio'],
                        ['superpoint_aachen', 'superglue'],
                        # [None, 'loftr_33_0.4'],
                        # [None, 'loftr_33_0.4_hc'],
                        # [None, 'loftr_25_0.5'],
                        # [None, 'loftr_25_0.5_hc'],
                        [None, 'loftr'],
                        [None, 'loftr_23_0.5'],
                        [None, 'loftr_23_0.5_hc'],
                        ]

    pipeline = ReconstructionPipeline(data_path=data_path, 
                                      output_path=output_path, 
                                      source_images_path=source_images_path,
                                      initial_models_path=initial_models_path,
                                      image_poses_file_name=image_poses_file_name,
                                      experiment_name=experiment_name, 
                                      extractor_matchers=extractor_matchers,
                                      pairs_dict=pairs,
                                      use_previous_as_ref=True
                                      )
    
    #RB, RT, LT, LB, covering the central field
    # polygon_corners = [(57.9431,34.3998), (82.5981,66.5854), (46.6873,95.0473), (21.6404,62.4076)] # 2018
    # polygon_corners = [(95.2749,4.1106), (119.8873,36.7558), (83.6157,65.8016), (59.0033,33.1364)] # 2019
    polygon_corners = [(91.7206,9.5767), (113.6489,38.3808), (83.6796,62.1366), (62.0337,34.1231)] # 2019 tight
    # polygon_corners = [(141.9008,71.4771), (163.0563,106.1057), (128.6143,133.3518), (106.9661,98.6574)] # 2020
    # polygon_corners = [(140.9325,75.7563), (162.8246,103.7484), (132.9247,127.5854), (110.0208,99.6374)] # 2020 tight
    minimum_distance = 1.7*1.97 # ~ 100 images per timestamp

    # pipeline.generate_poses(polygon_corners, minimum_distance)
    pipeline.localize_cameras(translation_error_thres=1.0,
                                  rotation_error_thres=3.0,
                                  ground_dist_thres=1.0)
