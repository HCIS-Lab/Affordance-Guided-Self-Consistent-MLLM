# Mitigating Cross-Modal Distraction and Ensuring Geometric Feasibility via Affordance-Guided, Self-Consistent MLLMs for Food Preparation Task Planning

[[paper](https://hcis-lab.github.io/Affordance-Guided-Self-Consistent-MLLM/static/pdfs/paper.pdf)] [[arXiv](https://arxiv.org/abs/2503.13055)] [[website](https://hcis-lab.github.io/Affordance-Guided-Self-Consistent-MLLM/)] 

![teaser](static/images/teaser.png)


<!-- ## Components
- Teaser video
- Images Carousel
- Youtube embedding
- Video Carousel
- PDF Poster
- Bibtex citation -->

<!-- ## System Requirements
- Linux (Teseted on Ubuntu 18.04)
- Python 3 (Tested on Python 3.7)
- Torch (Tested on Torch 1.9.1)
- Cuda (Tested on Cuda 11.4)
- GPU (Tested on Nvidia RTX3090)
- CPU (Tested on Intel COre i7-10700) -->

## Setup
- Install [Isaac Gym](https://developer.nvidia.com/isaac-gym/download) & Create Conda Environment
```sh
conda activate rlgpu
```
- Clone This Repo
```sh
git clone https://github.com/HCIS-Lab/Affordance-Guided-Self-Consistent-MLLM.git
```
- Install needed package.
1. Please check the [website](https://pytorch.org/get-started/previous-versions/) to install pytorch according to your local device.
3. Run pip install -r requirements.txt to install other package.
```sh
cd Affordance-Guided-Self-Consistent-MLLM
pip install -r requirements.txt
```

## Usage
- Set API key of OpenAI
```sh
export OPENAI_API_KEY=XXXXX
```

- Run experiment of different pipelines and task types
```sh
chmod +x run_experiment.sh
run_experiment.sh -n <PIPELINE_NAME> [ -e <EXP_ID>] [-c <CONFIG_FILE] [-l <LOG_ROOT>] [-t <MAX_TRIALS>]
```

```sh
# Run our method
run_experiment.sh -n our
```

- Collect trajectories of skills or manually control the robot
```sh
python data_collection.py
```

## TODO
- [ ] Upload requirement.txt
- [ ] Upload experimental log

## Acknowledgments
The work is sponsored by the National Science and Technology Council (NSTC) under grants 113-2813-C-A49-019-E. 

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Citation
```
@misc{shen2025mitigatingcrossmodaldistractionensuring,
      title={Mitigating Cross-Modal Distraction and Ensuring Geometric Feasibility via Affordance-Guided, Self-Consistent MLLMs for Food Preparation Task Planning}, 
      author={Yu-Hong Shen and Chuan-Yu Wu and Yi-Ru Yang and Yen-Ling Tai and Yi-Ting Chen},
      year={2025},
      eprint={2503.13055},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.13055}, 
}
```
