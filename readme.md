Official code:   
[ARES](https://zenodo.org/record/5088971)  
[ARES-specific adaptation of E3NN](https://zenodo.org/record/5090151)

# ARES implement
This is a trial implementation of ARES: [Geometric deep learning of RNA structure](https://www.science.org/doi/full/10.1126/science.abe5650)


## Run
```bash
# train
python train.py --device cuda 
# predict
python predict.py --model_path ARES.pt
```