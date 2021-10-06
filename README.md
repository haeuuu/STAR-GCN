# STAR-GCN
* pytorch & dgl implementation of Stacked and Reconstructed GCN for Recommender Systems
* paper : [STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems](https://arxiv.org/pdf/1905.13129.pdf)
  
</br>
  
### ✅ **ML-100K**
#### **Usage**
```
python main.py --data_name=ml-100k train --iteration=2000 --in_feats_dim=32
```
#### **Results**
```
Best iter : 1400
Best valid RMSE : 0.9013
Best test RMSE : 0.9150
```
```
reported RMSE : 0.8950
```
  
</br>
  
### ✅ **ML-1M**
#### **Usage**
```
python main.py --data_name=ml-1m train --iteration=2000 --in_feats_dim=64
```
#### **Results**
```
Best iter : 1990
Best valid RMSE : 0.8565
Best test RMSE : 0.8547
```
```
reported RMSE : 0.833
```
  
</br>
  
### **Notes**
* only transductive rating prediction is available
  
</br>
  
### **TODO**
- [ ] implement inductive rating prediction
- [ ] implement masked learning
- [ ] mini-batch learning
- [ ] sample-and-remove training
