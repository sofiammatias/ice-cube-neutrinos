# Ice Cube Neutrino Challenge

![image](https://res.cloudinary.com/icecube/image/upload/c_crop,g_south,h_500,w_2000/v1667419670/Header_HomeA_2000.jpg)

The goal of this competition from Kaggle is to predict/estimate **neutrino particle's direction** by developing a model based on data from the "IceCube" detector, which observes the cosmos from deep within the South Pole ice. This work could help scientists better understand exploding stars, gamma-ray bursts, and cataclysmic phenomena involving black holes, neutron stars and the fundamental properties of the neutrino itself.

Researchers have developed multiple approaches over the past ten years to reconstruct neutrino events. However, problems arise as existing solutions are far from perfect. They're either fast but inaccurate or more accurate at the price of huge computational costs. The **IceCube Neutrino Observatory** is the first detector of its kind, encompassing a cubic kilometer of ice and designed to search for the nearly massless neutrinos. An international group of scientists is responsible for the scientific research that makes up the IceCube Collaboration.

By making the process faster and more precise, it's possible to improve the reconstruction of neutrinos. As a result, we could gain a clearer image of our universe.

When detection events can be localized quickly enough, traditional telescopes are recruited to investigate short-lived neutrino sources such as supernovae or gamma ray bursts. Because the sky is huge better localization will not only associate neutrinos with sources but also to help partner observatories limit their search space. With an average of **three thousand events per second** to process, it's difficult to keep up with the stream of data using traditional methods. The main challenge in this competition is to quickly and accurately process a large number of events.

This competition used a hidden test set of roughly one million events, split between multiple batches.

I've ranked #501 in this challenge but that was only because I've simply run the benchmark notebook, which gave a score of 1.018, granting anyone a place in the learderboard. I would probably have ranked lower **near #770 place** with my simple neural network only (this is where people with a model score of 1.55 ranked).  

See more about the Ice Cube Neutrino Observatory [in this paper](https://arxiv.org/abs/1612.05093). 
See more about the Kaggle challenge [in this link](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/overview).

# Objectives

The main objective of this challenge is to predict a neutrino's direction in a timely manner, processing millions of rows in a short period of time. Two challenges arise: to properly treat sensors data to accurately predict a neutrino direction; to handle Gb's of data using parquet files.

To perform reconstruction tasks at neutrino telescopes, research has shown some limitations in both machine learning models and CNN models, especially has they can't handle the whole energy spectrum and the 6-node inputs in an efficient way (see [this paper](https://iopscience.iop.org/article/10.1088/1748-0221/17/11/P11003/pdf) to know more). Graph neural networks (GNNs) have been [the most promising solution so far](https://arxiv.org/abs/2210.12194), improving efficiency at different levels compared with previous models used before. 
GNN developed by [GraphNet](https://github.com/graphnet-team/graphnet) was the "benchmark model" for this challenge. 
**Azumith** and **zenith** angles were the outputs to be predicted.

The evaluation function for this challenge was the **mean angular error** between the predicted and true event origins:

```python
def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse 
    cosine (arccos) thereof is then the angle between the two input vectors
    
    Parameters:
    -----------
    
    az_true : float (or array thereof) 
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian
    
    Returns:
    --------
    
    dist : float
        mean over the angular distance(s) in radian
    '''
    
    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")
    
    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod =  np.clip(scalar_prod, -1, 1)
    
    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))
```

For each ```event_id```, ```azimuth``` and ```zenith``` must be predicted. Each ```event_id``` contained roughly 150 sensor measures, and each ```batch``` file contained 200 000 events. Each batch was saved in one \*.parquet file, and sensors data in a .csv file. It was required to merge all information together to list all data from all events.

Read more about the challenge description [here](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice).

# Results - Data Pipelines for Big Data

This challenge was a true personal challenge for me in many ways. I ended up focusing in learning more about the data pipeline, and how to deal with big data, to build my skills as a data engineer. Such huge amount of data, stored in .parquet files, required strategies to train the model without running our of RAM. There were several great solutions presented in several notebooks during the challenge. 

I was able to attempt at using **Dask** package and **Polars**, to replace Pandas. My experience with both is that Dask doesn't really work if you're using only one machine, being better used for parallel processing. **Polars** is a great package to replace Pandas in larger datasets, where your learning curve is fast, once you get the hang of the definition of lazy dataframes. 

This [kaggle notebook](https://www.kaggle.com/code/sofiamatias/icecube-predictions-simple-nn) generates submissions for one batch using a pre-trained simple neural network model. I was glad I received a score at all!! (I've scored 1.55, which was not a good score). This meant that I had a sucessful submission: my notebook was able to read the test dataset of one million entries and predict values from it using my model. In this challenge, has it is happening in prize kaggle challenges, the test dataset was hidden and you could only submit the code/notebook. So, it was a blind implementation. And a sucessful one!!

# The Model

I've tried to use LGBM and simple regression neural networks for this challenge. True to [investigations previously done](https://iopscience.iop.org/article/10.1088/1748-0221/17/11/P11003/pdf), these solutions didn't give any good results.

I was able to run the benchmark notebook, which used GraphNet GNN, and do some small adjustments. However, I took a lot of time in learning Dask, Polars and training other models, and I didn't have time to study the solution with GNN's and implement alternatives.

In the end, I was quite satisfied with the results, simply training models and being able to handle with such big dataset sucessfully, while learning about tools such as Polars. I've felt quite out of my league in this challenge (even worse than in Learning Equality), and was about to give up several times during the challenge. Only cheer persistence and the opportunity to train/handle with big data led me to enter the leaderboard. And that made me feel good.
