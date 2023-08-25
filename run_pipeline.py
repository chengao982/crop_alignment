import os
from generate_image_poses import ImagePoseGenerator
from reconstructor import Reconstruction
from localizer import CameraLocalization
from evaluator import Evaluation
import time
import shutil

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
            print(f"Generating image_poses.txt for subfolder {subfolder}\n")

            generator = ImagePoseGenerator(os.path.join(self.data_path, subfolder), 'image_poses.txt')
            generator.process_camera_poses(polygon_corners, distance_threshold=minimum_distance)

    def build_inital_models(self):
        start_time = time.time()
        for idx, subfolder in enumerate(self.subfolders):
            print(f"Running intial reconstruction for subfolder {subfolder}...\n")
            
            if not os.path.isfile(os.path.join(output_path, 'done.txt')):
                output_path = os.path.join(self.initial_models_output_path, subfolder)
                data_path = os.path.join(self.data_path, subfolder)
                source_images_path = os.path.join(self.source_images_path, subfolder, 'RAW/JPEG')

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
                with open(os.path.join(output_path, 'done.txt'), 'w') as f:
                    f.write('done')

            else:
                print(f'Initial model of {subfolder} already exists\n')

        end_time = time.time()
        run_time = end_time - start_time
        print(f"Initial Reconstruction Runtime: {run_time}\n")

    def _localize_cameras(self, extractor, matcher):
        previous_data_ref_path = None
        previous_reconstruction_ref_path = None

        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print(f"Running localization for subfolder {subfolder}...\n")

            identifier = extractor if extractor else matcher
            output_path = os.path.join(self.output_path, identifier, subfolder)
            data_temp_path = os.path.join(self.data_path, subfolder)
            reconstruction_temp_path = os.path.join(self.initial_models_output_path, subfolder, 'sparse/aligned')

            if self.use_previous_as_ref and previous_reconstruction_ref_path is not None:
                data_ref_path = previous_data_ref_path
                reconstruction_ref_path = previous_reconstruction_ref_path
            else:
                data_ref_path = os.path.join(self.data_path, self.subfolders[0])
                reconstruction_ref_path = os.path.join(self.output_path, identifier, self.subfolders[0], 'sparse/corrected')

            if idx == 0:
                shutil.rmtree(reconstruction_ref_path, ignore_errors=True) # Remove the existing destination folder if it exists
                shutil.copytree(reconstruction_temp_path, reconstruction_ref_path) # Copy the entire folder from source to destination

                previous_data_ref_path = data_ref_path
                previous_reconstruction_ref_path = reconstruction_ref_path

            else:
                if not os.path.isfile(os.path.join(output_path, 'loc_done.txt')):
                    plotting = True if (idx==1 or idx==len(self.subfolders)-1) else False
                    localizer = CameraLocalization(output_path=output_path,
                                                images_ref_path=os.path.join(data_ref_path, 'images4reconstruction'),
                                                images_temp_path=os.path.join(data_temp_path, 'images4reconstruction'),
                                                reconstruction_ref_path=reconstruction_ref_path,
                                                reconstruction_temp_path=reconstruction_temp_path,
                                                extractor=extractor,
                                                matcher=matcher,
                                                gps_noise=self.gps_error,
                                                dist_threshold=self.distance_threshold,
                                                plotting=plotting)
                    localizer.run()

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

    def localize_cameras(self):
        for extractor_matcher in self.extractor_matchers:
            extractor, matcher = extractor_matcher
            print(f"Running localization with {extractor}/{matcher} ...\n")
            self._localize_cameras(extractor, matcher)

    def _evaluate(self, extractor, matcher):
        for subfolder in self.subfolders:
            start_time = time.time()
            print(f"Running evaulation for subfolder {subfolder}...\n")

            if not os.path.isfile(os.path.join(output_path, 'eval_done.txt')):
                identifier = extractor if extractor else matcher
                output_path = os.path.join(self.output_path, identifier, subfolder)
                data_gt_path = os.path.join(self.data_path, subfolder)
                reconstruction_path = os.path.join(output_path, 'sparse/corrected')

                evaluator = Evaluation(data_gt_path=data_gt_path,
                                    output_path=output_path,
                                    reconstruction_path=reconstruction_path,
                                    translation_error_thres=self.translation_error_thres,
                                    rotation_error_thres=self.rotation_error_thres,
                                    ground_dist_threshold=self.ground_dist_threshold)
                evaluator.run()

                # done flag
                with open(os.path.join(output_path, 'eval_done.txt'), 'w') as f:
                    f.write('done')
            else:
                print(f'Evaluation for {subfolder} has already been done\n')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"{identifier} Evaulation Runtime for {subfolder}: {run_time}\n")

    def evalate(self, translation_error_thres=1.0, rotation_error_thres=2.0, ground_dist_threshold=0.2):
        self.translation_error_thres = translation_error_thres
        self.rotation_error_thres = rotation_error_thres
        self.ground_dist_threshold = ground_dist_threshold

        for extractor_matcher in self.extractor_matchers:
            extractor, matcher = extractor_matcher
            print(f"Running evaluation for {extractor}/{matcher} ...\n")
            self._evaluate(extractor, matcher)
        
if __name__ == "__main__":
    # data_path = '/Volumes/Plextor/crops'
    # output_path = '/Volumes/Plextor/output'
    # source_images_path = '/Volumes/Plextor/crops'
    data_path = '/home/gao/dataset_loftr/crop/real'
    output_path = '/home/gao/crop_alignment/output'
    source_images_path = '/mnt/buzz_newhd/home/v4rl/pheno-datasets'

    extractor_matchers = [
                        ['sift', 'NN-ratio'],
                        ['superpoint_max', 'superglue'],
                        [None, 'loftr'],
                        [None, 'loftr_33_0.4'],
                        [None, 'loftr_33_0.4_hc'],
                        [None, 'loftr_25_0.5'],
                        [None, 'loftr_25_0.5_hc'],
                        ]
    experiment_name = 'version_0'

    pipeline = ReconstructionPipeline(data_path=data_path, 
                                      output_path=output_path, 
                                      source_images_path=source_images_path,
                                      experiment_name=experiment_name, 
                                      extractor_matchers=extractor_matchers,
                                      use_previous_as_ref=True
                                      )
    
    polygon_corners = [(57.9431,34.3998), (82.5981,66.5854), (46.6873,95.0473), (21.6404,62.4076)] #RB, RT, LT, LB, covering the central field
    minimum_distance = 1.7*1.97 # ~ 100 images per timestamp
    # pipeline.generate_poses(polygon_corners, minimum_distance)
    pipeline.build_inital_models()
    pipeline.localize_cameras()
    pipeline.evalate()
