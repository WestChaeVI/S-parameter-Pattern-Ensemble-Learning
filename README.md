# S-parameter Pattern Ensemble Learning(2024)      

**Chaejun Seo<sup>1,o</sup>**, Dahyun Lee<sup>1</sup>, Minkyu Kang<sup>2</sup>, Namgyeong Kim<sup>1</sup>, Hyunwoo Nam<sup>2</sup>, Taeyeob Kang<sup>2,\*</sup>, Seunghyeok Hong<sup>1,*</sup>    
Ensemble Learning of Scattering Coefficient Patterns for Non-destructive Estimation of Copper Corrosion Severity in Electronic Products    
[한국정보과학회 학술발표논문집, In Proceeding (unable not yet)]()       


------------------------------------------------------------------------------------------------------------------------       

## Abstract     

The normalization of robotic electronic products highlights the importance of safety management. The corrosion of metals within electronic packages can lead to malfunction, posing a risk to human safety. Therefore, a more accurate estimation of corrosion severity is necessary. However, conventional visual inspections often face difficulties in obtaining a sufficient field of view to confirm the state, and there are limitations in that corrosion levels often need to be verified destructively. In this study, we propose a method for estimating the severity of copper corrosion using tree-based models and the scattering parameter (S-parameter) pattern. Through this method, we aim to quantitatively assess corrosion levels as continuous parameters using non-destructive means. This will help prevent unexpected system or component failures and estimate the appropriate timing for maintenance. All experiments were conducted with an average of 30 results, and various interpretations of the results, such as how efficiently the dataset can be used and which combination and cycle provide better estimation accuracy, are provided.      


------------------------------------------------------------------------------------------------------------------------       


## Introduction     

The normalization of electronic products, which has expanded into the domain of robots operating in close proximity to humans, has underscored the importance of safety management. Corrosion occurring within the electronic packages of products that operate in wide ranges or near the human body can lead not only to inconvenience due to malfunctions but also poses a risk of causing physical harm to humans through unexpected malfunctions (Figure 1). Moreover, in harsh operating environments such as high temperature, humidity, and pressure, there is a high possibility of corrosion of metallic materials within electronic packages. This deteriorates the intrinsic characteristics of electronic products, affecting the overall device performance and leading to operational issues. Corrosion is not just a simple defect but one of the progressive failures. Furthermore, it can occur during the operational phase even if not detected during the manufacturing process. Therefore, it is crucial to quantitatively assess the continuous progression of corrosion.      

### Figure 1      
#### Severity of corrosion occurring within electronic packages embedded in safety-critical products    

<p align="center"><img src="https://github.com/WestChaeVI/S-parameter-Pattern-Ensemble-Learning/assets/104747868/74b03306-f0e1-4d84-a183-76fa33409d396" height='200'>     

This study utilized tree-based ensemble learning methods, including XGBoost, LightGBM, and NGBoost, to develop estimation models for the severity of metal corrosion. In this paper, combinations of four features (S11, S21, Ph11, Ph21) were created, and corrosion severity estimation was performed for each combination. Additionally, we present results obtained by further subdividing the frequency band for estimation, considering efficiency. Furthermore, we propose grouping cycles of corrosion severity based on the measured dates to determine which cycle was easier to estimate.    

------------------------------------------------------------------------------------------------------------------------       

## Related Work      

### S-parameter     
Scattering parameters are the most widely used circuit response parameters in RF (Radio Frequency) applications, representing the ratio of the output voltage to the input voltage in the frequency domain (Figure 2). $𝑉_1^+$ and $𝑉_1^−$, $𝑉_2^+$ and $𝑉_2^−$ represent the incident voltage waves and reflections at each input port (Port 1) and output port (Port 2), respectively. These values can be expressed as scattering parameters of the two-port network (Equation 1).      

#### Figure 2      
##### Two port network of S-parameters with copper    

<p align="center"><img src="https://github.com/WestChaeVI/S-parameter-Pattern-Ensemble-Learning/assets/104747868/28f2e93d-8a2d-450c-a97d-2db2e9221d20" height='150'>        

#### Equation 1      

$$V_1^-= S_{11}V_1^+ + S_{12}V_2^+,$$      

$$V_2^-= S_{21}V_1^+ + S_{22}V_2^+,$$      

$$S_{11}=(V_1^-)/(V_1^+)  \ when \ V_2^+=0,$$      

$$S_{21}=(V_2^-)/(V_1^+)  \ when \  V_2^+=0,$$      

$$S_{12}=(V_1^-)/(V_2^+)  \ when \  V_1^+=0,$$      

$$S_{22}=(V_2^-)/(V_2^+)  \ when \  V_1^+=0,$$      

$S_{11}$ represents the signal reflected back to Port 1 after passing through the copper terminal at Port 1, indicating reflection loss. $S_{21}$ represents the signal transmitted to Port 2 after passing through the copper terminal at Port 1, indicating transmission loss. If specific patterns of scattering parameters indicate certain characteristics of the reliability of the copper terminal, learning to recognize these patterns can help prevent unexpected failures and anticipate the appropriate timing for maintenance.     

### Prior Research     


Machine learning methodologies for estimating the corrosion rate of metals have been continuously researched with various approaches. For example, a study combined particle swarm optimization and support vector regression (SVR) approaches with five parameters (temperature, dissolved oxygen, salinity, pH, oxidation-reduction potential) to design a model for estimating the corrosion rate of 3C steel in various seawater environments. The estimation results were compared with those of a BPNN (Back Propagation Neural Network) model. Additionally, corrosion rates were estimated by utilizing the corrosion characteristics of aluminum-silicon (AlSi) alloys with different stirring speeds and pH values in various media containing silicon (Si). The study proposed a method for building an optimal model using various types of MRLO (Machine Regression Learner Optimization) techniques.       
In this paper, we propose a model for estimating corrosion using tree-based regression models based on information from copper terminals. Furthermore, we compare the results with those of the BPNN model and SVR model from previous studies and present the results of comparisons with three models: MLP (Multi-Layer Perceptron), CNN (Convolutional Neural Network), and ViT (Vision Transformer).      

------------------------------------------------------------------------------------------------------------------------       


## Experimental design    

### Dataset      
The dataset used in the experiment consists of a total of 280 data points recorded for 5 consecutive days for each of the 56 electronic components. The data were recorded from 0.0003GHz to 14GHz in increments of 0.07GHz. Each data point includes 201 values of $S_{11}$, $S_{21}$, $Ph_{11}$, and $Ph_{21}$, along with the corrosion severity. $Ph_{11}$ and $Ph_{21}$ represent the phase of $S_{11}$ and $S_{21}$, respectively. The ground truth labels are the corrosion severity (%) visually confirmed by destructively opening the electronic packages.

For each electronic component, cycles were examined based on the measurement date (Figure 3, 4). A cycle was defined as exposing the components to a high humidity (95% R.H.) environment for 5 days, increasing the chamber temperature from 30°C to 60°C for 8 hours, then gradually decreasing it back to 30°C over the next 16 hours.

The evaluation metrics used to quantitatively compare the designed dataset and models are widely used regression metrics: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R-squared score (Equations 2). The dataset was split into a training set of 188 data points and a test set of 92 data points using a train-test split ratio of 0.33 with a random state of 42. The scores obtained from a total of 30 experiments were averaged.

#### Equation 2      

$$ \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} {(y_i - \hat{y}_i)^2} } $$     

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} {|y_i - \hat{y}_i|} $$     

$$ \text{R}\_2 = 1 - \frac{\sum_{i=1}^{n} {(y_i - \hat{y}\_i)^2}}{\sum_{i=1}^{n} {(y\_i - \bar{y})^2}} $$     

#### Figure 3      
##### $S_{11}$, $S_{21}$, $Ph_{11}$, $Ph_{21}$ for copper terminals by cycle average measurement    

<p align="center"><img src="https://github.com/WestChaeVI/S-parameter-Pattern-Ensemble-Learning/assets/104747868/fbcbbaa1-c3cb-49b9-95b2-c6e958b2c468" height='400'>   


#### Figure 4      
##### Bar graph of average corrosion severity by cycle    

<p align="center"><img src="https://github.com/WestChaeVI/S-parameter-Pattern-Ensemble-Learning/assets/104747868/08decef3-f22f-4a91-bc3f-9fd42d17e10d" height='300'>     


### Comparison experiment method using efficient datasets      

The designed dataset comprised 15 combinations of $S_{11}$, $S_{21}$, $Ph_{11}$, and $Ph_{21}$. The frequency range was divided into 201 values at 1/4 intervals, and the dataset was accumulated for experimentation. Subsequently, to focus on how efficiently time and resources could be utilized, the first quarter of the frequency range was further divided into 1/7 units, and the same cumulative experiment was conducted. When the frequency range was divided into quarters, the number of columns in the table data for each combination increased in order of the number of combinations: 50, 100, 150, and 200.      


------------------------------------------------------------------------------------------------------------------------       

## Results and Discussion      

In the preliminary study, artificial neural networks were implemented. However, based on the average MAE on the test set, MLP showed a performance of 12.186, CNN showed 10.678, and ViT showed 10.024, which were lower compared to ensemble techniques. Generally, when considering one combination, XGBoost showed the best performance with MAE of 9.351, and when considering four combinations, its performance improved to 7.970. Additionally, LightGBM showed the best performance with MAE of 8.546 for two combinations, and NGBoost showed the best performance with MAE of 8.187 for three combinations (Table 1).

The best performing combination and model were $S_{11}S_{21}Ph_{11}Ph_{21}$ and XGBoost with an average MAE of 7.617. When considering one combination, the highest performance was achieved by using only 50% of the dataset. For two and three combinations, using 75% of the dataset resulted in the highest performance, and for four combinations, using 100% of the dataset was optimal.

When 25% and 75% of the dataset were used, the combination of $S_{21}$ and $Ph_{11}$ showed the best performance with average MAE of 8.136 and 8.004, respectively. When 50% and 100% of the dataset were used, the combination of $S_{11}S_{21}Ph_{11}Ph_{21}$ showed the best performance with average MAE of 7.798 and 7.667, respectively.

Efficiency-focused experiments yielded the following results:

+ For $S_{11}$, using only up to the 4/7 point resulted in an average MAE of 9.240, which was similar to using only up to the 1/4 point, and LightGBM showed 0.181 lower MAE.     
+ For $S_{21}$, using only up to the 6/7 point resulted in an average MAE of 8.634, which was similar to using only up to the 1/4 point, and LightGBM showed 0.165 lower MAE.      
+ Lastly, for $S_{11}Ph_{11}$, using only up to the 4/7 point resulted in an average MAE 0.121 lower than using only up to the 1/4 point, and NGBoost showed 0.086 lower MAE (Table 2).       


### Comparative analysis of experiment results by cycle      

Based on the experimental results, the cycles were grouped, and the performance of each cycle was examined. Cycle 1 showed the lowest average MAE of 5.612 across the three models (Table 3). While the actual corrosion severity average difference between cycles was approximately 19%, the MAE in this study was at a level of 9%. This indicates that non-destructive machine learning techniques can play a role in detecting the severity of corrosion in electronic packages and ensuring safety.     

Conventional impedance detection methods typically require severe corrosion to the extent that the current connection is interrupted before detection. In contrast, the application of non-destructive machine learning techniques allows for early detection of corrosion occurring in the early stages between cycles 1 and 2. Therefore, it is proposed as a technology that can contribute to the improvement of safety in actual electronic products.      

------------------------------------------------------------------------------------------------------------------------        


#### Table 1      
##### Model performance average(std) MAE standard result comparison table for each combination       

| Combination | XGB | LGBM | NGB | Avg std |
| --- | --- | --- | --- | --- |
| 1 | 9.351 | 9.539 | 9.658 | 9.516 (0.155) |
| 2 | 8.624 | 8.546 | 8.568 | 8.579 (0.040) |
| 3 | 8.193 | 8.211 | 8.187 | 8.197 (0.012) |
| 4 | 7.970 | 8.037 | 7.994 | 8.000 (0.034) |     


#### Table 2     
##### Cumulative experiment results for measuring dataset efficiency for each major combination (average of XGB, LGBM, NGB)       

| Max Freq. | 3.5GHz | 7GHz | 10.5GHz | 14GHz |
| --- | --- | --- | --- | --- |
| **S11** | | | | |
| RMSE | 11.656 (0.083) | 11.625 (0.062) | 11.717 (0.162) | 11.707 (0.121) |
| MAE | 9.201 (0.050) | 9.152 (0.055) | 9.099 (0.129) | 9.232 (0.057) |
| R2 | 0.824 (0.003) | 0.827 (0.002) | 0.824 (0.005) | 0.819 (0.004) |
| **S21** | | | | |
| RMSE | 11.288 (0.037) | 10.882 (0.258) | 11.177 (0.193) | 11.076 (0.189) |
| MAE | 8.570 (0.068) | 8.293 (0.176) | 8.591 (0.098) | 8.360 (0.053) |
| R2 | 0.838 (0.001) | 0.852 (0.007) | 0.841 (0.006) | 0.840 (0.006) |
| **S11Ph11** | | | | |
| RMSE | 11.657 (0.102) | 11.356 (0.121) | 11.110 (0.169) | 11.545 (0.147) |
| MAE | 9.188 (0.087) | 8.922 (0.130) | 8.704 (0.122) | 9.033 (0.106) |
| R2 | 0.820 (0.003) | 0.831 (0.003) | 0.840 (0.005) | 0.821 (0.005) |

| Max Freq. | 0.5GHz | 1GHz | 1.5GHz | 2GHz | 2.5GHz | 3GHz |
| --- | --- | --- | --- | --- | --- | --- |
| **S11** | | | | | | |
| RMSE | 12.296 (0.043) | 12.059 (0.046) | 11.869 (0.032) | 11.703 (0.130) | 11.806 (0.112) | 11.983 (0.052) |
| MAE | 9.663 (0.036) | 9.468 (0.030) | 9.372 (0.031) | 9.240 (0.110) | 9.265 (0.079) | 9.438 (0.041) |
| R2 | 0.801 (0.002) | 0.808 (0.002) | 0.814 (0.001) | 0.823 (0.004) | 0.814 (0.003) | 0.810 (0.002) |
| **S21** | | | | | | |
| RMSE | 13.178 (0.198) | 12.377 (0.071) | 12.065 (0.129) | 11.466 (0.211) | 11.626 (0.085) | 11.493 (0.237) |
| MAE | 10.114 (0.086) | 9.479 (0.090) | 9.188 (0.060) | 8.696 (0.184) | 8.790 (0.071) | 8.634 (0.201) |
| R2 | 0.779 (0.007) | 0.795 (0.002) | 0.810 (0.004) | 0.826 (0.006) | 0.821 (0.002) | 0.828 (0.007) |
| **S11Ph11** | | | | | | |
| RMSE | 12.353 (0.053) | 11.565 (0.063) | 11.838 (0.014) | 11.561 (0.093) | 11.580 (0.111) | 11.773 (0.151) |
| MAE | 9.815 (0.063) | 9.210 (0.038) | 9.145 (0.060) | 9.067 (0.042) | 9.124 (0.046) | 9.252 (0.073) |
| R2 | 0.797 (0.001) | 0.834 (0.002) | 0.818 (0.001) | 0.829 (0.003) | 0.825 (0.003) | 0.823 (0.004) |       

#### Table 3     
##### Model performance average MAE standard result comparison table by cycle       


| Cycle | XGB | LGBM | NGB | Avg |
| --- | --- | --- | --- | --- |
| 1 | 5.527 | 5.726 | 5.583 | 5.612 |
| 2 | 9.726 | 9.527 | 9.717 | 9.657 |
| 3 | 9.978 | 10.081 | 9.847 | 9.969 |
| 4 | 9.569 | 9.451 | 9.392 | 9.471 |
| 5 | 9.981 | 9.824 | 10.071 | 9.959 |     
