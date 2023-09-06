import os
from generate_image_poses import ImagePoseGenerator
from reconstructor import Reconstruction
from localizer import CameraLocalization
from evaluator import Evaluation
import time
import shutil
import warnings

class ReconstructionPipeline:
    def __init__(self, 
                 data_path, 
                 output_path, 
                 source_images_path, 
                 experiment_name, 
                 extractor_matchers,
                 gps_error=5.0, 
                 distance_threshold=0.20, 
                 use_previous_as_ref=False
                 ):
        
        self.data_path = data_path
        self.output_path = os.path.join(output_path, experiment_name)
        self.source_images_path = source_images_path
        self.initial_models_output_path = os.path.join(self.output_path, 'initial_models')
        self.extractor_matchers = extractor_matchers
        self.plot = True
        self.gps_error = gps_error
        self.distance_threshold = distance_threshold
        self.use_previous_as_ref = use_previous_as_ref
        subfolders = next(os.walk(self.data_path))[1]
        self.subfolders = sorted(subfolders)

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

            generator = ImagePoseGenerator(os.path.join(self.data_path, subfolder), 'image_poses.txt')
            generator.process_camera_poses(polygon_corners, distance_threshold=minimum_distance)

        print('====================Pose File Generation Done====================\n')

    def build_inital_models(self):
        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print('--------------------Intial Reconstruction--------------------')
            print(f"Running intial reconstruction for subfolder {subfolder}...\n")
            
            output_path = os.path.join(self.initial_models_output_path, subfolder)
            data_path = os.path.join(self.data_path, subfolder)
            source_images_path = os.path.join(self.source_images_path, subfolder, 'RAW/JPEG')

            if not os.path.isfile(os.path.join(output_path, 'rec_done.txt')):

                if idx == 0:
                    print("Running reconstructor ground truth model ...\n")
                    reconstructor = Reconstruction(data_path=data_path, 
                                                output_path=output_path, 
                                                source_images_path=source_images_path,
                                                error=0.0)
                    reconstructor.run()

                else:
                    print(f"Running reconstructor temporal model {idx} ...\n")
                    reconstructor = Reconstruction(data_path=data_path, 
                                                output_path=output_path, 
                                                source_images_path=source_images_path,
                                                error=self.gps_error)
                    reconstructor.run()

                # done flag
                with open(os.path.join(output_path, 'rec_done.txt'), 'w') as f:
                    f.write('done')

            else:
                print(f'Initial model of {subfolder} already exists\n')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"Initial Reconstruction Runtime for {subfolder}: {run_time}\n")

        print('====================Intial Reconstruction Done====================\n')

    def evalate_reconstruction(self, translation_error_thres, rotation_error_thres, ground_dist_thres):
        for subfolder in self.subfolders:
            start_time = time.time()
            print('-----------------Reconstruction Evaluation-----------------')
            print(f"Running evaulation for subfolder {subfolder}...")

            output_path = os.path.join(self.initial_models_output_path, subfolder)
            data_gt_path = os.path.join(self.data_path, subfolder)
            reconstruction_path = os.path.join(output_path, 'sparse/aligned')

            if not os.path.isfile(os.path.join(output_path, 'eval_done.txt')):
                evaluator = Evaluation(data_gt_path=data_gt_path,
                                    output_path=output_path,
                                    reconstruction_path=reconstruction_path,
                                    translation_error_thres=translation_error_thres,
                                    rotation_error_thres=rotation_error_thres,
                                    ground_dist_thres=ground_dist_thres)
                evaluator.run()

                # done flag
                with open(os.path.join(output_path, 'eval_done.txt'), 'w') as f:
                    f.write('done')
            else:
                print(f'Evaluation for {subfolder} has already been done\n')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"Reconstruction Evaulation Runtime for {subfolder}: {run_time}\n")

        print('====================Reconstruction Evaluation Done====================\n')

    def _localize_cameras(self, extractor, matcher):
        print(f'==========Start localization with {extractor} / {matcher}==========\n')

        previous_data_ref_path = None
        previous_reconstruction_ref_path = None

        identifier = extractor if extractor else matcher

        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print(f'-----------------{identifier} Localization-----------------')
            print(f"Running localization for subfolder {subfolder}...")

            output_path = os.path.join(self.output_path, identifier, subfolder)
            data_temp_path = os.path.join(self.data_path, subfolder)
            reconstruction_temp_path = os.path.join(self.initial_models_output_path, subfolder, 'sparse/0') #aligned

            if self.use_previous_as_ref and previous_reconstruction_ref_path is not None:
                data_ref_path = previous_data_ref_path
                reconstruction_ref_path = previous_reconstruction_ref_path
            else:
                data_ref_path = os.path.join(self.data_path, self.subfolders[0])
                reconstruction_ref_path = os.path.join(self.output_path, identifier, self.subfolders[0], 'sparse/corrected')

            if idx == 0:
                if not os.path.isfile(os.path.join(output_path, 'loc_done.txt')):
                    reconstruction_init_path = os.path.join(self.initial_models_output_path, subfolder, 'sparse/aligned')
                    shutil.rmtree(reconstruction_ref_path, ignore_errors=True) # Remove the existing destination folder if it exists
                    shutil.copytree(reconstruction_init_path, reconstruction_ref_path) # Copy the entire folder from source to destination
                    # done flag
                    with open(os.path.join(output_path, 'loc_done.txt'), 'w') as f:
                        f.write('done')

                previous_data_ref_path = data_ref_path
                previous_reconstruction_ref_path = reconstruction_ref_path

            else:
                if os.path.isfile(os.path.join(output_path, 'loc_failed.txt')):
                    if self.use_previous_as_ref:
                        print(f'Localization failed, abort localization for current & subsequent subfolders\n')
                        break
                    else:
                        print(f'Localization failed, abort localization for current subfolder\n')
                        continue

                if not os.path.isfile(os.path.join(output_path, 'loc_done.txt')):
                    localizer = CameraLocalization(output_path=output_path,
                                                images_ref_path=os.path.join(data_ref_path, 'images4reconstruction'),
                                                images_temp_path=os.path.join(data_temp_path, 'images4reconstruction'),
                                                reconstruction_ref_path=reconstruction_ref_path,
                                                reconstruction_temp_path=reconstruction_temp_path,
                                                extractor=extractor,
                                                matcher=matcher,
                                                gps_noise=self.gps_error,
                                                dist_threshold=self.distance_threshold,
                                                plotting=True)
                    localizer.run()

                    if localizer.is_successful is False:
                    # abort localization for subsequent subfolders if alignment failed. not enough inliers have been found
                        with open(os.path.join(output_path, 'loc_failed.txt'), 'w') as f:
                            f.write('failed')
                        if self.use_previous_as_ref:
                            print(f'Localization failed, abort localization for current & subsequent subfolders\n')
                            break
                        else:
                            print(f'Localization failed, abort localization for current subfolder\n')
                            continue
                    else:
                        # done flag
                        with open(os.path.join(output_path, 'loc_done.txt'), 'w') as f:
                            f.write('done')

                else:
                    print(f'Localization for {subfolder} has already been done\n')

                previous_data_ref_path = data_temp_path
                previous_reconstruction_ref_path = os.path.join(output_path, 'sparse/corrected')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"{identifier} Localization Runtime for {subfolder}: {run_time}\n")

        print(f'===================={identifier} Localization Done====================\n')

    def localize_cameras(self):
        for extractor_matcher in self.extractor_matchers:
            extractor, matcher = extractor_matcher
            self._localize_cameras(extractor, matcher)

    def _evalate_localization(self, extractor, matcher, translation_error_thres, rotation_error_thres, ground_dist_thres):
        print(f'==========Start evaluation for {extractor} / {matcher}==========\n')

        identifier = extractor if extractor else matcher

        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print(f'-----------------{identifier} Evaluation-----------------')
            print(f"Running evaulation for subfolder {subfolder}...")

            output_path = os.path.join(self.output_path, identifier, subfolder)
            data_gt_path = os.path.join(self.data_path, subfolder)
            reconstruction_path = os.path.join(output_path, 'sparse/corrected')

            if os.path.isfile(os.path.join(output_path, 'loc_failed.txt')):
                if self.use_previous_as_ref:
                    print(f'No localization found, abort evaluation for current & subsequent subfolders\n')
                    break
                else:
                    print(f'No localization found, abort evaluation for current subfolders\n')
                    continue

            if not os.path.isfile(os.path.join(output_path, 'eval_done.txt')):
                print('-----------------corrected_aligned-----------------')
                evaluator = Evaluation(data_gt_path=data_gt_path,
                                    output_path=output_path,
                                    reconstruction_path=reconstruction_path,
                                    translation_error_thres=translation_error_thres,
                                    rotation_error_thres=rotation_error_thres,
                                    ground_dist_thres=ground_dist_thres)
                evaluator.run()

                print('----------------localized------------------')
                if idx!= 0:
                    evaluator.run_localized()

                # done flag
                with open(os.path.join(output_path, 'eval_done.txt'), 'w') as f:
                    f.write('done')
            else:
                print(f'Evaluation for {subfolder} has already been done\n')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"{identifier} Evaulation Runtime for {subfolder}: {run_time}\n")

        print(f'===================={identifier} Evaluation Done====================\n')

    def evalate_localization(self, translation_error_thres, rotation_error_thres, ground_dist_thres):
        for extractor_matcher in self.extractor_matchers:
            extractor, matcher = extractor_matcher
            self._evalate_localization(extractor, matcher, translation_error_thres, rotation_error_thres, ground_dist_thres)
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    # data_path = '/Volumes/Plextor/20190313_20190705_int16'
    # output_path = '/Volumes/Plextor/output'
    # source_images_path = '/Volumes/Plextor/crops'
    # data_path = '/home/gao/dataset_loftr/crop/real_first_month'
    # output_path = '/home/gao/crop_alignment/output'
    # source_images_path = '/mnt/buzz_newhd/home/v4rl/pheno-datasets'

    data_path = '/home/gao/dataset_loftr/crop/20190313_20190705_int16'
    output_path = '/home/gao/crop_alignment/output'
    source_images_path = '/mnt/usb-ROG_ESD-S1C_N5D0AP040191-0:0'

    experiment_name = '20190313_20190705_int16'

    extractor_matchers = [
                        # ['sift', 'NN-ratio'],
                        ['superpoint_max', 'superglue'],
                        # [None, 'loftr_33_0.4'],
                        # [None, 'loftr_33_0.4_hc'],
                        # [None, 'loftr_25_0.5'],
                        # [None, 'loftr_25_0.5_hc'],
                        [None, 'loftr_23_0.5'],
                        [None, 'loftr_23_0.5_hc'],
                        [None, 'loftr'],
                        ]

    pipeline = ReconstructionPipeline(data_path=data_path, 
                                      output_path=output_path, 
                                      source_images_path=source_images_path,
                                      experiment_name=experiment_name, 
                                      extractor_matchers=extractor_matchers,
                                      use_previous_as_ref=True
                                      )
    
    #RB, RT, LT, LB, covering the central field
    # polygon_corners = [(57.9431,34.3998), (82.5981,66.5854), (46.6873,95.0473), (21.6404,62.4076)] # 2018
    polygon_corners = [(95.2749,4.1106), (119.8873,36.7558), (83.6157,65.8016), (59.0033,33.1364)] # 2019
    # polygon_corners = [(141.9008,71.4771), (163.0563,106.1057), (128.6143,133.3518), (106.9661,98.6574)] # 2020
    minimum_distance = 1.7*1.97 # ~ 100 images per timestamp

    pipeline.generate_poses(polygon_corners, minimum_distance)
    pipeline.build_inital_models()
    pipeline.localize_cameras()
    pipeline.evalate_reconstruction(translation_error_thres=1.0, 
                                rotation_error_thres=3.0, 
                                ground_dist_thres=1.0)
    pipeline.evalate_localization(translation_error_thres=1.0,
                                  rotation_error_thres=3.0,
                                  ground_dist_thres=1.0)
