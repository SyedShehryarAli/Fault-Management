"""Experiment for building ensemble and individual classifiers."""

from src.utils.arg_parser import *
from src.utils.directory_management import *
from src.data_processing.data_processing import *
from src.data_processing.data_resampling import *
from src.utils.save_utils import class_distribution, data_analysis
from src.models.ensemble_model import *
from src.inference.inference import binary_classification_inference, multi_classification_inference
from src.interpretation.interpret_model import *
from sklearn.metrics import confusion_matrix
import pandas as pd


def main(args=None):
    """Main function to set up the experiment."""
    arg_parser = get_parser(args)
    resampling_flag = arg_parser.resampling_flag
    resample_method = arg_parser.resampling_method.name
    train_test_ratio = arg_parser.test_ratio
    voting_method = arg_parser.voting_method
    class_replace_flag = arg_parser.class_replace_flag
    print(train_test_ratio,voting_method, resample_method)
    # Setting file structure for accessing data and storing results
    res_dir = ''
    data_file_dir = ''
    data_path = get_data_dir()
    # Load data using pandas dataframe 
    target_label = 'f_nv'
    df_data = pd.read_csv(data_path+'/combined_data_part1.csv')
    print(df_data[target_label].unique())
    # Initial pre-processing
    # Replace all of classes to one class belonging to fault types
    if class_replace_flag:
        unique_classes = df_data[target_label].unique()
        classes_to_replace = unique_classes[1:]
        df_data.loc[df_data[target_label].isin(classes_to_replace), target_label] = 1
    df_data[target_label] = df_data[target_label].astype('int64')
    print(df_data)
    # Data Spliting into training and testing samples
    X_train, X_test, y_train, y_test = get_data(df_data.copy(), target_label, train_test_ratio)
    # Normalization using min-max 
    X_train_normalized = normalize_data(X_train)
    X_test_normalized = normalize_data(X_test)
    # Shape 
    print(type(X_train_normalized), X_train_normalized.shape)
    print(type(y_train), y_train.shape)
    print(type(X_test_normalized), X_test_normalized.shape)
    print(type(y_test), y_test.shape)
    
    if resampling_flag:
        # Get resulting directory path 
        res_dir = get_result_dir('Ensemble Experiment', resample_method)
        data_file_dir = get_file_dir(res_dir, voting_method)
        X_train_normalized[target_label] = y_train
        # Class distribution before smote based re-sampling 
        class_distribution(X_train_normalized, target_label, 'without-resampled', data_file_dir)
        if resample_method=='SMOTE':
            X_train_normalized = smote_resampling(X_train_normalized, target_label)
        elif resample_method=='ADASYN':
            X_train_normalized = adasyn_resampling(X_train_normalized, target_label)
        # Class distribution after smote based re-sampling 
        class_distribution(X_train_normalized, target_label, resample_method, data_file_dir)
        y_train = X_train_normalized[target_label]
        X_train_normalized.drop(target_label, axis=1, inplace=True)
        # Shape 
        print(type(X_train_normalized), X_train_normalized.shape)
        print(type(y_train), y_train.shape)
        
    else:
        res_dir = get_result_dir('Ensemble Experiment', 'Imbalanced')
        data_file_dir = get_file_dir(res_dir, voting_method)
        df_train = X_train_normalized.copy()
        df_train[target_label] = y_train
        # Class distribution after smote based re-sampling 
        class_distribution(df_train, target_label, 'imbalanced', data_file_dir)
    
    # Correlation analysis of input features with respect to the output feature
    data_analysis(df_data, target_label, res_dir)
    # Train ensemble and inidiviudal learners
    print('Training...')
    ensemble_clf = ensemble_classifier(X_train_normalized, y_train, voting_method)
    individual_clf = individual_classifiers(X_train_normalized, y_train)
    # Evaluate ensmeble and individual learners 
    print('Inference...')
    if class_replace_flag:
        binary_classification_inference(X_test_normalized, y_test, ensemble_clf, individual_clf, data_file_dir)
    else:
        multi_classification_inference(X_test_normalized, y_test, ensemble_clf, individual_clf, data_file_dir)
    # Interpret predictions made by ensmeble and individual learners 
    print('Interpretation...')
    feature_names = df_data.columns.to_list()
    #ensemble_classifier_interpretation(ensemble_clf, X_train_normalized, X_test_normalized, feature_names, voting_method, data_file_dir)
    

if __name__ == '__main__':
    main()