#!/usr/bin/env nextflow
nextflow.enable.dsl=2


process train_Hoptimus0 {
    container 'elirn98/cancer_subtyping_nextflow_pipeline:v1'
    
    input: 
    val model_name
    val save_path
    val data_path
  
    output:
    val("/app/${params.SAVE_PATH}/model.pth.tar"), emit: hoptimus0_model
    val("/app/${params.SAVE_PATH}/train_statistics.pkl"), emit: model_statistics

    script:
    """
    python3 /app/train.py --model $model_name --csv_path /data --save_dir /app/$save_path

    """
}

process test_Hoptimus0 {
    container 'elirn98/cancer_subtyping_nextflow_pipeline:v1'

    input: 
    val model_path
    val save_path
    val data_path

    output:
    val("/app/${params.SAVE_PATH}/test_results.pkl"), emit: test_results

    script:
    """
    echo $model_path
    python3 /app/evaluate_model.py --model_path $model_path --csv_path /data  --save_dir /app/$save_path
    """
}
 process visualize{
    container 'elirn98/cancer_subtyping_nextflow_pipeline:v1'

    input: 
    val history_path
    val result_path
    val save_plot_path

    script:
    """
    echo $history_path
    echo $result_path
    python3 /app/visualize.py --history_path $history_path --result_path $result_path --save_plot_path /app/$save_plot_path
    """
}

workflow {

    (hoptimus0_model, model_statistics)=train_Hoptimus0(params.MODEL, params.SAVE_PATH, params.DATA_PATH)
    test_results = test_Hoptimus0(hoptimus0_model, params.SAVE_PATH, params.DATA_PATH)
    visualize(model_statistics, test_results, params.PLOT_PATH)


}