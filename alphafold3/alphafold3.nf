/*
Created on 12/8/24
Last Modified on 02/23/26
@author: Kameron Bates
*/
// 
params.peptides = null // csv with no header of peptide and corresponding hla
params.out_dir = null
params.alphafold_sif = null // path to alphafold3 container
params.alphafold_model = null //path to alhphafold 3 model
params.alphafold_database = null //path to alhphafold 3 databsae
params.python_sif = null // path to python container
params.hlaDict = null // dictionary of MHC MSAs


process generate_json {
    queue 'compute'
    executor "slurm"
    tag "$peptide"
    clusterOptions '--time=0:30:00'
    publishDir "${params.out_dir}/$hla/json", mode: 'copy'

    input:
    tuple val(peptide), val(hla)

    output:
    tuple val(peptide), val(hla), path("*.json")

    script:
    """
    module load singularity

    singularity exec --cleanenv \
    ${params.python_sif} \
    python generate_alphafold_json.py \
    -p $peptide 

    """
 }

process data_pipeline {
    queue 'compute'
    cpus '8'
    memory '80GB'
    executor "slurm"
    clusterOptions '--time=2:00:00'
    errorStrategy 'ignore'
    tag "$peptide"

    input:
    tuple val(peptide), val(hla), path(json)

    output:
    tuple val(peptide), val(hla), path('*/*.json')

    script:
    """
    module load singularity

    singularity exec --cleanenv \
    ${params.alphafold_sif} \
    python /app/alphafold/run_alphafold.py \
    --json_path=$json \
    --model_dir=${params.alphafold_model} \
    --db_dir=${params.alphafold_database} \
    --output_dir=. \
    --norun_inference

    """
}

process add_hla {
    queue 'compute'
    memory '12GB'
    executor "slurm"
    clusterOptions '--time=0:30:00'
    errorStrategy 'ignore'
    tag "$peptide"

    input: 
    tuple val(peptide), val(hla), path(peptide_json, stageAs: "?/*")

    output:
    tuple val(peptide), val(hla), path("*.json")

    script:
    """
    module load singularity

    singularity exec --cleanenv \
    ${params.python_sif} \
    python add_hla_json.py -j $peptide_json --hla $hla \
    --hlaDictionar ${params.hlaDict} -p $peptide

    """


}


process alpha_inference {
    queue 'gpu-a100'
    cpus '8'
    clusterOptions '--nodes=1 --ntasks=1 --gres=gpu:1 --time=0:45:00'
    memory { 24.GB * task.attempt }
    executor "slurm"
    errorStrategy 'retry'
    maxRetries 3
    tag "$peptide"
    publishDir "${params.out_dir}/$hla/confidences/", mode: 'copy'

    input:
    tuple val(peptide), val(hla), path(msa_json)

    output:
    tuple val(peptide), val(hla), path('*')

    script:
    """
    module load singularity

    singularity exec --nv --cleanenv \
    ${params.alphafold_sif} \
    python /app/alphafold/run_alphafold.py \
    --json_path=$msa_json \
    --model_dir=${params.alphafold_model} \
    --db_dir=${params.alphafold_database} \
    --output_dir=. \
    --norun_data_pipeline

    """
}


workflow {
    channel
    .fromPath(params.peptides)
    .splitCsv()
    .map { row -> tuple(row[0], row[1]) }
    .set { json }

    json_single = generate_json(json)
    
    pipeline_json = data_pipeline(json_single)
    hla_data = add_hla(pipeline_json)

    alpha_inference(hla_data)
    
}