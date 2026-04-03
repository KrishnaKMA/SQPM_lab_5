# SOFE3980U – Lab 5: Data Quality and Validation
**Student:** Krishna Mallick  
**Student ID:** 100876443

---

## Task 1 – Great Expectations

### Three Expectations Used and Their Relevance

**Expectation 1: `ExpectColumnValuesToBeBetween` on `pedestrianLocationX_TopLeft` (min=0, max=900)**

This expectation validates that the X-coordinate of the top-left corner of every detected pedestrian bounding box falls within a pixel range of 0 to 900. The Labels dataset records pedestrian positions captured by a camera with a fixed resolution (approximately 854 × 720 pixels). Any X value outside this range would indicate a corrupted or out-of-bounds detection — either a sensor error, a faulty annotation, or a placeholder value that was not properly filtered. This is essential for downstream computer vision tasks where out-of-range bounding box coordinates would crash rendering or model inference pipelines.

**Expectation 2: `ExpectColumnValuesToNotBeNull` on `Timestamp`**

This expectation ensures that every row in the dataset has a valid, non-null timestamp. In a time-series dataset collected from a simulation environment — where Car 1, Car 2, and the pedestrian are tracked across frames — the timestamp is the primary key for temporal ordering and multi-sensor synchronization. A missing timestamp makes it impossible to correctly associate a frame's image files with its coordinate data, breaking any replay or analysis pipeline that depends on chronological ordering. Validating completeness here is a foundational data quality check.

**Expectation 3: `ExpectColumnValuesToMatchRegex` on `Ground_Truth_View` (regex: `C_\d{3}\.png`)**

This expectation checks that every entry in the `Ground_Truth_View` column matches the naming pattern `C_XXX.png` (e.g., `C_001.png`, `C_085.png`). The dataset uses a structured three-file naming convention per frame (A = occluded view, B = occluding car view, C = ground truth). If any filename deviates from this pattern — due to a file rename, manual editing error, or data corruption — it would prevent the correct ground truth image from being loaded during model evaluation. Regex validation acts as a lightweight schema check on unstructured string columns, catching naming inconsistencies early.

---

## Task 2 – CleanLab: Mislabeled Data

### Question: Why might this data point be mislabeled, and which feature values could have caused the misclassification?

The Iris dataset contains three classes: Setosa (0), Versicolor (1), and Virginica (2). In the notebook, `np.random.seed(42)` is used before randomly selecting 5 indices and assigning them incorrect labels. CleanLab's `CleanLearning` wrapper uses cross-validation to estimate the probability of each class for every training sample and then flags samples where the assigned label has a low predicted probability — indicating the model consistently predicts a different class for that sample.

A data point is likely mislabeled when its feature values are characteristic of one species but its label says it belongs to another. The most common cause in the Iris dataset is overlap between Versicolor and Virginica, which share similar ranges for petal length (4.5–5.1 cm for Versicolor vs. 4.9–6.9 cm for Virginica) and petal width (1.0–1.8 cm vs. 1.4–2.5 cm). A Versicolor sample with a petal length near 5.0 cm and petal width near 1.8 cm sits at the boundary between the two classes. If such a sample was randomly relabeled as Virginica, the Random Forest classifier — trained on the remaining, predominantly correct data — would consistently predict Versicolor for it, causing CleanLab to flag it as a label issue.

Sepal dimensions (sepal length and sepal width) are less discriminative between Versicolor and Virginica and would contribute less to the misclassification detection. However, for Setosa samples (which are linearly separable from the other two classes), any random mislabeling would be immediately obvious because Setosa's feature values (small petals: ~1.4 cm length, ~0.2 cm width) are far outside the range of the other two species. CleanLab would flag these with very high confidence.

In summary, the feature values most responsible for detected misclassifications are **petal length** and **petal width**, since these provide the clearest separation between classes and make label errors most detectable by the model.

---

## Task 3 – CleanLab: Anomaly Detection

### Question 1: Do these suspected anomalous data points match what you expect for their species? Why or why not?

No, the suspected anomalous data points do not match what would be expected for their species. The notebook artificially introduces anomalies by setting `petal length (cm)` to random values between 5 and 7 cm for 10 randomly selected samples. In the original Iris dataset:

- **Setosa** has petal lengths of approximately **1.0–1.9 cm**
- **Versicolor** has petal lengths of approximately **3.0–5.1 cm**
- **Virginica** has petal lengths of approximately **4.9–6.9 cm**

Any Setosa or Versicolor sample with a petal length of 5–7 cm would be far outside its species' expected range. A Setosa with petal length of 6.0 cm, for example, does not resemble Setosa at all — it resembles Virginica. Even for Virginica samples, values pushed to the high end (6.5–7 cm) would be at or beyond the outer edge of the natural distribution. CleanLab detects these because the classifier, trained on the largely unmodified dataset, cannot reconcile the anomalous petal length value with the given species label, flagging the sample as a likely label issue or anomaly.

### Question 2: Which feature (sepal length, petal length, etc.) seems most unusual in these points?

**Petal length** is clearly the most unusual feature in the detected anomalous points. This is by design — the notebook explicitly modifies only `petal length (cm)` by replacing it with a uniform random value between 5 and 7 cm. All other features (sepal length, sepal width, petal width) retain their original, species-consistent values. Petal length is also the single most discriminative feature in the Iris dataset for distinguishing between species, so artificially inflating it creates the strongest signal for anomaly detection. The other features would appear normal for the reported species, making petal length stand out as the sole dimension that does not fit.

### Question 3: How can you check if these values are truly anomalies using the original dataset?

There are several ways to verify whether the flagged values are genuine anomalies:

1. **Statistical bounds check:** Compute the mean and standard deviation of each feature per species in the original (unmodified) Iris dataset. Flag any sample where a feature value falls more than 2–3 standard deviations from its species mean. For example, Setosa petal length has a mean of ~1.46 cm and std of ~0.17 cm, so any Setosa sample with petal length above ~1.97 cm (mean + 3σ) is a statistical outlier.

2. **Distribution comparison:** Plot box plots or violin plots of petal length per species from the original dataset, then overlay the flagged sample values. Any value visually outside the whiskers of its species' distribution is a candidate anomaly.

3. **Nearest-neighbor check:** For each flagged sample, find the K nearest neighbors in feature space from the original unmodified dataset. If those neighbors predominantly belong to a different species than the flagged sample's label, the value is likely anomalous or mislabeled.

4. **Cross-reference with the original index:** Since `np.random.seed(42)` was used, the anomaly indices are deterministic. You can retrieve the original (pre-modification) petal length values for those specific indices from the unloaded `iris.data` array and compare them to the modified values in `df`, directly confirming which values were altered.
