# FOCAL

- torch 1.9.0
- scikit-learn 1.2.1
- torch_kmeans 0.2.0

## Usage

- For training:
```bash
python main.py --type='train'
```

- For testing:
```bash
python main.py --type='test_single'
```
FOCAL will detect the images in the `demo/input/` and save the results in the `demo/output/` directory.

- For prepare the training/test datasets:
```bash
python main.py --type='flist' --path_input 'demo/input/' --path_gt 'demo/gt/' --nickname 'demo'
```

**Note: The pretrained FOCAL can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12ayIO9PU4wvqWqniT3KtH8tCvrZ-M-zd?usp=sharing).**

**Note: To facilitate reproduction, we provide the download link of test datasets used in the FOCAL experiments: [Columbia, Coverage, CASIA, NIST](https://drive.google.com/file/d/1jbVCBc4ofj3oASKEOtxEtuAw6cBl8wfv/view?usp=sharing), [MISD](https://drive.google.com/file/d/17acVu8Tal7pYu0cEeRQ8EBu6AIK188MG/view?usp=sharing), [FFpp](https://drive.google.com/file/d/13FTwLZjO9QUiwpH7-iRFKAlkKQGRjWYy/view?usp=sharing). Please follow relevant licenses and use them for research purposes only.**

## Citation

If you use this code for your research, please citing the reference:
```
@article{focal,
  title={Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering},
  author={H. Wu and Y. Chen and J. Zhou},
  year={2023}
}
```
