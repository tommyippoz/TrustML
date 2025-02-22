# SPROUT - a Safety wraPper thROugh ensembles of UncertainTy measures

Python Framework to improve safety of classifiers by computing quantitative uncertainty in their predictions

## Aim/Concept of the Project

SPROUT implements quantitative uncertainty/confidence measures and integrates well with existing frameworks (e.g., Pandas, Scikit-Learn, PYOD, AutoGluon, and many more) that are commonly used in the machine learning domain for classification. 

While designing, implementing and testing such library we made sure it would work with supervised classifiers, as well as unsupervised classifiers. Also, we created connectors for tabular datasets as well as image datasets such that those classifiers can be fed with different inputs and provide confidence measures related to the execution of many classifiers on datasets with a different structure

## Uncertainty Measures

This work focuses on uncertainty measures that are not classifier-specific, but instead have a generic formula-tion that pairs well with any classifier, which is seen as a black-box. The framework implements uncertainty measures UM1 to UM9, which process at least one of: i) input data dp, ii) class prediction dp_prob. Importantly, all measures but UM2, UM3 and UM8 require training data for set-up, and all measures but UM2, UM3, UM4 are parametric, meaning that different values of parameters may be employed to craft different instances of the same measure.
Uncertainty measures are implemented in SPROUT through the abstract class [UncertaintyCalculator](sprout/UncertaintyCalculator.py) and derivatives

###	UM1: Confidence Interval (UncertaintyCalculator.ConfidenceInterval)
A confidence interval defines the statistical distribution underlying the value of a feature and thus provides a range, constrained to the parameter 0 ≤ w ≤ 1, in which  feature values are expected to fall. The confidence level w represents the long-run proportion of feature values (at the given confidence level) that theoretically contain the true value of the feature. UM1 measures how many feature values falls inside their confidence interval. The higher the UM1, the more feature values of dp are outside their confidence interval, which indicates high uncertainty in the prediction.

### UM2: Maximum Likelihood (UncertaintyCalculator.MaxProbUncertainty)
Given dp_prob produced by a classifier for a given dp, we identify UM2 as the maximum probability of dp_prob. The higher the UM2, the more uncertain the output of the classifier.

### UM3: Entropy of Probabilities (UncertaintyCalculator.EntropyUncertainty) 
We retrieve the dp_prob produced by a classifier for a given dp and we compute UM3 using db_prob entropy. The higher the UM3, the more uncertain the classifier: a dp_prob array with constant values (i.e., all classes have the same probability) generates the highest UM3 of 1.

### UM4: Bayesian Uncertainty (UncertaintyCalculator.ExternalSupervisedUncertainty) 
This measure uses a Naïve Bayes process to estimate the probability that the input data point dp belongs to each of the possible c classes. Briefly, this process applies Bayes' theorem assuming strong (i.e., naive) independence between the features. As such, UM4 may not apply to many classification problems, especially those dealing with images, where a pixel (feature) clearly depends on its surrounding pixels.

### UM5: Combined Uncertainty (UncertaintyCalculator.CombinedUncertainty) 
UM5 uses a classifier chk_c that acts as a checker of the main classifier clf. UM5 has positive sign if clf and chk_c agree on the predicted class, negative otherwise. The absolute value of UM5 is quantified according to the entropy (UM3) in the results of chk_c. UM5 ranges from -1 to 1. UM5 = 1 translates to high confidence that the prediction of clf is correct, UM5 = -1 means high confidence that the prediction is a misclassification, letting UM5 = 0 show maximum uncertainty.

### UM6: Multi-Combined Uncertainty (UncertaintyCalculator.MultiCombinedUncertainty) 
UM6 computes uncertainty relying on more than one checker. UM6 uses a set CC of ncc checking classifiers, computes UM5 for each chk_c ∈ CC  with respect to clf, and averages the results. The more checking classifiers in CC agree with clf, the higher the UM6.

### UM7: Feature Bagging (UncertaintyCalculator.FeatureBaggingUncertainty) 
UM7 exploits the concept of bagging, a method for generating multiple versions of a classifier bagC: each instance of bagC is trained using different subsets of the original training set, and decides using restricted knowledge. Should classifiers predict different classes for a given data point dp, UM7 would have low value and predictions should be treated with high uncertainty.

### UM8: Neighbor Agreement (UncertaintyCalculator.NeighborsUncertainty) 
UM8 finds the k nearest neighbors of a data point dp. Then, it classifies dp and its k neighbors using clf: the more neighbors are assigned to the same class predicted for dp, the higher the UM8. The lower the value, the more disagreement in classifying neighboring data points to dp. This means that the input data point dp lies in an unstable region of the input space, which translates to high uncertainty (low UM8) in the prediction.

### UM9 Reconstruction Loss (UncertaintyCalculator.ReconstructionLoss) 
Reconstruction loss quantifies to what extent the input data point is an unseen, out-of-distribution data point, and as such it is likely to generate misclassifications. We compute UM9 through the reconstruction error of autoencoders, which are unsupervised neural networks composed of different layers to learn efficient encodings of the input data. A low UM9 value instead indicates that dp belongs to an expected distribution and as such is like-ly to be correctly classified.

## Dependencies

SPROUT needs the following libraries:
- <a href="https://numpy.org/">NumPy</a>
- <a href="https://scipy.org/">SciPy</a>
- <a href="https://pandas.pydata.org/">Pandas</a>
- <a href="https://scikit-learn.org/stable/">SKLearn</a>

## Usage

SPROUT can wrap any classifier you may want to use, provided that the classifier implements scikit-learn like interfaces, namely
- classifier.predict(test_set): takes a 2D ndarray and returns an array of predictions for each item of test_set
- classifier.predict_proba(test_set): takes a 2D ndarray and returns a 2D ndarray where each line contains probabilities for a given data point in the test_set

Assuming the classifier has such a structure, a SPROUT analysis with three calculators can be set up as it can be seen in the `examples` folder

[Simple sample](https://github.com/tommyippoz/SPROUT/blob/9d03c8b41e514c4bc0875f1304e5a6b684b889c4/examples/simple_example.py#L1-L23)

## Citation and Credits

Zoppi, T., Ceccarelli, A., & Bondavalli, A. (2023). Ensembling Uncertainty Measures to Improve Safety of Black-Box Classifiers. In ECAI 2023 (pp. 3156-3164). IOS Press.

Paper is available at https://ebooks.iospress.nl/doi/10.3233/FAIA230635

Developed @ University of Florence and University of Trento, Italy

Contributors
- Tommaso Zoppi
- Fahad Ahmed Khokhar
- Leonardo Bargiotti
