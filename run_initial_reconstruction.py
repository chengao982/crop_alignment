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
                 image_poses_file_name, 
                 experiment_name, 
                 gps_error=5.0, 
                 ):
        
        self.data_path = data_path
        self.output_path = os.path.join(output_path, experiment_name)
        self.source_images_path = source_images_path
        self.image_poses_file_name = image_poses_file_name
        self.gt_models_output_path = os.path.join(self.output_path, 'gt_models')
        self.noisy_models_output_path = os.path.join(self.output_path, 'noisy_models')
        self.gps_error = gps_error
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

            generator = ImagePoseGenerator(os.path.join(self.data_path, subfolder), self.image_poses_file_name)
            generator.process_camera_poses(polygon_corners, distance_threshold=minimum_distance)

        print('====================Pose File Generation Done====================\n')

    def build_gt_models(self):
        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print('--------------------GT Reconstruction--------------------')
            print(f"Running intial reconstruction for subfolder {subfolder}...\n")
            
            output_path = os.path.join(self.gt_models_output_path, subfolder)
            data_path = os.path.join(self.data_path, subfolder)
            source_images_path = os.path.join(self.source_images_path, subfolder, 'RAW/JPEG')

            if not os.path.isfile(os.path.join(output_path, 'rec_done.txt')):
                print(f"Running reconstructor ground truth model {idx} ...\n")
                reconstructor = Reconstruction(data_path=data_path, 
                                            output_path=output_path, 
                                            source_images_path=source_images_path,
                                            image_poses_file_name=self.image_poses_file_name,
                                            error=0.0)
                reconstructor.run()

                # done flag
                with open(os.path.join(output_path, 'rec_done.txt'), 'w') as f:
                    f.write('done')

            else:
                print(f'GT model of {subfolder} already exists\n')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"GT Reconstruction Runtime for {subfolder}: {run_time}\n")

        print('====================GT Reconstruction Done====================\n')

    def build_noisy_models(self):
        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print('-------------------- Noisy Reconstruction--------------------')
            print(f"Running intial reconstruction for subfolder {subfolder}...\n")
            
            output_path = os.path.join(self.noisy_models_output_path, subfolder)
            data_path = os.path.join(self.data_path, subfolder)
            source_images_path = os.path.join(self.source_images_path, subfolder, 'RAW/JPEG')

            if not os.path.isfile(os.path.join(output_path, 'rec_done.txt')):
                print(f"Running reconstructor temporal model {idx} ...\n")
                reconstructor = Reconstruction(data_path=data_path, 
                                            output_path=output_path, 
                                            source_images_path=source_images_path,
                                            image_poses_file_name=self.image_poses_file_name,
                                            error=self.gps_error)
                reconstructor.run()

                # done flag
                with open(os.path.join(output_path, 'rec_done.txt'), 'w') as f:
                    f.write('done')

            else:
                print(f'Noisy model of {subfolder} already exists\n')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"Noisy Reconstruction Runtime for {subfolder}: {run_time}\n")

        print('====================Noisy Reconstruction Done====================\n')

    def evalate_reconstruction(self, translation_error_thres, rotation_error_thres, ground_dist_thres):
        for subfolder in self.subfolders:
            start_time = time.time()
            print('-----------------Reconstruction Evaluation-----------------')
            print(f"Running evaulation for subfolder {subfolder}...")

            output_path = os.path.join(self.gt_models_output_path, subfolder)
            data_gt_path = os.path.join(self.data_path, subfolder)
            reconstruction_path = os.path.join(output_path, 'sparse/aligned')

            if not os.path.isfile(os.path.join(output_path, 'eval_done.txt')):
                evaluator = Evaluation(data_gt_path=data_gt_path,
                                    output_path=output_path,
                                    reconstruction_path=reconstruction_path,
                                    image_poses_file_name=self.image_poses_file_name,
                                    translation_error_thres=translation_error_thres,
                                    rotation_error_thres=rotation_error_thres,
                                    ground_dist_thres=ground_dist_thres)
                evaluator.run()

                # done flag
                with open(os.path.join(output_path, 'eval_done.txt'), 'w') as f:
                    f.write('done')
            else:
                print(f'Evaluation for {subfolder} has already been done\n')



            output_path = os.path.join(self.noisy_models_output_path, subfolder)

            if not os.path.isfile(os.path.join(output_path, 'eval_done.txt')):
                evaluator = Evaluation(data_gt_path=data_gt_path,
                                    output_path=output_path,
                                    reconstruction_path=reconstruction_path,
                                    image_poses_file_name=self.image_poses_file_name,
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

    experiment_name = 'crop_2019'

    image_poses_file_name = 'image_poses_tight.txt'

    pipeline = ReconstructionPipeline(data_path=data_path, 
                                      output_path=output_path, 
                                      source_images_path=source_images_path,
                                      image_poses_file_name=image_poses_file_name,
                                      experiment_name=experiment_name, 
                                      )
    
    #RB, RT, LT, LB, covering the central field
    # polygon_corners = [(57.9431,34.3998), (82.5981,66.5854), (46.6873,95.0473), (21.6404,62.4076)] # 2018
    # polygon_corners = [(95.2749,4.1106), (119.8873,36.7558), (83.6157,65.8016), (59.0033,33.1364)] # 2019
    polygon_corners = [(91.7206,9.5767), (113.6489,38.3808), (83.6796,62.1366), (62.0337,34.1231)] # 2019 tight
    # polygon_corners = [(141.9008,71.4771), (163.0563,106.1057), (128.6143,133.3518), (106.9661,98.6574)] # 2020
    # polygon_corners = [(140.9325,75.7563), (162.8246,103.7484), (132.9247,127.5854), (110.0208,99.6374)] # 2020 tight
    minimum_distance = 1.7*1.97 # ~ 100 images per timestamp

    pipeline.generate_poses(polygon_corners, minimum_distance)
    pipeline.build_gt_models()
    pipeline.build_noisy_models()
    pipeline.evalate_reconstruction(translation_error_thres=1.0, 
                                rotation_error_thres=3.0, 
                                ground_dist_thres=1.0)
