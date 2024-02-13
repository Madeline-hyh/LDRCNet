import os
from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderMPR,DataLoaderEnhance,DataLoaderMyTrain

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)

def get_mytraining_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderMyTrain(rgb_dir, img_options)

def get_enhanced_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderEnhance(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)

def get_mpr_data(in_dir,out_dir, img_options):
    assert os.path.exists(in_dir)
    assert os.path.exists(out_dir)
    return DataLoaderMPR(in_dir,out_dir, img_options)