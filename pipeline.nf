#!/usr/bin/env nextflow
//nextflow.enable.dsl=2


process load_and_preprocess_data {
    conda 'environment.yml'
    debug true

    input: 
    val COVID_PATH
    val NONCOVID_PATH
    val intermediate_result_path
    val data_plots
  
    output:
    val("${PWD}/${intermediate_result_path}/x_train.npy"), emit: x_train
    val("${PWD}/${intermediate_result_path}/y_train.npy"), emit: y_train
    val("${PWD}/${intermediate_result_path}/x_test.npy"), emit: x_test
    val("${PWD}/${intermediate_result_path}/y_test.npy"), emit: y_test

    script:
    """
    mkdir -p ${PWD}/${params.PROCESSED_DATA_PATH}
    mkdir -p ${PWD}/${params.PLOT_PATH}
    python $PWD/load_data_and_preprocess.py --covid_positive_path $PWD/$COVID_PATH --covid_negative_path $PWD/$NONCOVID_PATH --intermediate_result_path $PWD/$intermediate_result_path --data_plots $PWD/$data_plots
    """
}

process train_VGG16 {
    conda 'environment.yml'
    debug true

    input: 
    val x_train
    val y_train
    val x_test
    val y_test
    val path
  
    output:
    val("${PWD}/${params.VGG16}/VGG16_CT_model.h5"), emit: VGG16_model
    //val("${PWD}/${params.VGG16}/VGG16_CT_model_wights.h5"), emit: VGG16_weights
    val("${PWD}/${params.VGG16}/trainHistoryDict"), emit: VGG16_history

    script:
    """
    mkdir -p ${PWD}/${params.VGG16}
    python $PWD/train_VGG16.py --x_train $x_train --y_train $y_train --x_test $x_test --y_test $y_test --path  $PWD/$path
    """
}

process predict_and_save_results{
    conda 'environment.yml'
    debug true

    input: 
    val x_test
    val y_test
    val model_path
    val history_path
    val prediction_result_path

    output:
    val("${PWD}/${params.VGG16}/prediction_results"), emit: prediction_results

    script:
    """
    python $PWD/evaluate_and_predict.py --x_test $x_test --y_test $y_test --model_path $model_path --history_path $history_path --prediction_result_path $PWD/$prediction_result_path
    """
}

process visualize_results{
    conda 'environment.yml'
    debug true

    input: 
    val history_path
    val result_path
    val save_plot_path

    script:
    """
    python $PWD/visualize.py --history_path $history_path --result_path $result_path --save_plot_path $PWD/$save_plot_path
    """
}


workflow {
    (ch_x_train, ch_y_train,ch_x_test,ch_y_test )=load_and_preprocess_data(params.COVID_PATH, params.NONCOVID_PATH, params.PROCESSED_DATA_PATH, params.PLOT_PATH)
    (ch_VGG16_model , ch_VGG16_history) = train_VGG16(ch_x_train, ch_y_train, ch_x_test, ch_y_test, params.VGG16)
    ch_prediction_results = predict_and_save_results(ch_x_test, ch_y_test, ch_VGG16_model, ch_VGG16_history, params.VGG16)
    visualize_results(ch_VGG16_history, ch_prediction_results, params.VGG16)
     
}