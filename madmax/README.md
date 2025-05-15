# MADMax: Distributed Machine Learning Model Acceleration

This repository is for open-sourcing of the International Symposium on Computer Architecture (ISCA) 2024 paper **MAD Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems**.

## Setup
```
# Run the setup script
./setup.sh

# Create and activate conda environment
conda create -n madmax python=3.9
conda activate madmax

# Install dependencies
pip install -r requirements.txt
```

## Running Examples

### DLRM Example
```
python run_model.py
```

### LLM Example
```
python run_model.py --model-cfg-file 'model_cfgs/llm/llama2_70b.json' \
                    --system-cfg-file 'system_cfgs/dc_a/dc_a_2048.json' \
                    --task-cfg-file 'task_cfgs/llm/llm_train.json'
```

Successful runs will display output ending with `**************************************************`.

## Repository Structure
- `model_cfgs/`: Model architecture configurations
- `models/`: Model implementation code
- `system_cfgs/`: Distributed system configurations
- `systems/`: System implementation code
- `task_cfgs/`: Task configurations
- `tasks/`: Task execution workload descriptions
- `run_model.py`: Main simulation entry point

## Citation
Please cite [our ISCA'24 paper](https://arxiv.org/abs/2310.02784) as:

``` bibtex
@INPROCEEDINGS{hsia2024madmax,
  author={Hsia, Samuel and Golden, Alicia and Acun, Bilge and Ardalani, Newsha and DeVito, Zachary and Wei, Gu-Yeon and Brooks, David and Wu, Carole-Jean},
  booktitle={2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)}, 
  title={MAD-Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems}, 
  year={2024},
  volume={},
  number={},
  pages={818-833},
  doi={10.1109/ISCA59077.2024.00064}
}
```
