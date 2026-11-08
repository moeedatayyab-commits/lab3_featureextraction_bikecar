# Lab 03 Report: Turning Text and Images into Numbers

**Course:** Introduction to Artificial Intelligence (AI-101L)

---

## 1. What I did

Raw text and raw images cannot go straight into a traditional machine learning model, because the model needs **numbers in a fixed-length vector**. In this lab I did two transformations:

- **Part A (text):** Bag-of-Words with `CountVectorizer`. Documents become a document-term matrix.
- **Part B (images):** resize, grayscale and flatten. Images become a 1-D vector of pixel values.

In both cases the data is made simpler, and **some information is thrown away**. Naming exactly what is lost was the main point of the lab.

---

## 2. Part A: Bag-of-Words

**Corpus (3 documents):**
1. The sun is shining today.
2. The weather is good, the sun is great.
3. A sunny day is a wonderful day.

**Result:** a matrix of shape **(3, 11)**. The 11 words (features) are: `day, good, great, is, shining, sun, sunny, the, today, weather, wonderful`.

| | day | good | great | is | shining | sun | sunny | the | today | weather | wonderful |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Doc 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| Doc 2 | 0 | 1 | 1 | 2 | 0 | 1 | 0 | 2 | 0 | 1 | 0 |
| Doc 3 | 2 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |

**Two silent decisions of `CountVectorizer`** (both must be stated):
1. It **lowercases** every word, so "The" and "the" are the same feature.
2. Its token pattern `(?u)\b\w\w+\b` needs **2 or more characters**, so the word **"a"** is dropped. I checked this in code: `'a' in feature_names` is `False`.

**Sparsity:** 18 of the 33 cells are zero, which is **54.5%**. For a real corpus of thousands of documents it is above 99%, which is why scikit-learn returns a sparse matrix.

**Note:** in the lab sheet's "expected output" table the columns `wonderful` and `weather` are in the wrong order. The real alphabetical order is `weather`, then `wonderful`, as above.

---

## 3. Part B: Image to feature vector

Pipeline: **load → resize → grayscale → flatten**.

- An RGB image of size H x W has **H x W x 3** features.
- After grayscale it has **H x W** features (3 times fewer).
- After flattening it is a 1-D vector of length **H x W**.
- The function `image_to_features(path, size=(64, 64))` (in `image_utils.py`) does all four steps. It always returns a vector of length **4,096** for a 64 x 64 size, whatever the original image size.

| Item | Value |
|---|---|
| My sample image (`bike/bike_01.jpg`), size H x W x C | 333 x 500 x 3 |
| RGB features | 333 x 500 x 3 = **499,500** |
| Grayscale features / flattened length | 333 x 500 = **166,500** (3 times fewer) |
| After resize to 64 x 64 | 4,096 |
| After resize to 32 x 32 | 1,024 |

---

## 4. Discussion questions (all seven)

### Part A

**1. What does each column in the matrix represent?**
Each column is one **unique word** (a "term") from the vocabulary of the whole corpus. The number in a column is **how many times that word appears in that document**. Here there are 11 columns, so each document becomes a vector of 11 numbers.

**2. What is the machine-understandable format?**
It is the **document-term matrix**: a table of numbers with one row per document and one column per word (here 3 x 11). Every document, whatever its length, becomes a **fixed-length vector of counts**. A model can do maths on numbers, but not on raw sentences.

**3. How is this an oversimplification of the text?**
- **Word order** is lost: "dog bites man" and "man bites dog" give the same vector.
- **Grammar and meaning** are lost: the model does not know that "sun" and "sunny" are related.
- **Capital letters** are removed, and **one-letter words** such as "a" and **punctuation** are dropped.
- All words count equally, so common words like "is" and "the" can dominate.
- The matrix is mostly zeros, so it wastes space.

### Part B

**1. How does grayscale conversion act as feature engineering?**
A colour image has three channels (red, green, blue) per pixel. Grayscale replaces them with **one brightness value**, so the number of features becomes **3 times smaller**. It keeps shapes, edges and brightness, and **throws away colour**. The result is a simpler input, less memory, faster training and less risk of overfitting. It is a good trade when colour does not matter (reading digits) and a bad one when colour is the clue (ripe vs unripe fruit).

**2. If the image is H x W, how long is the flattened vector?**
**H x W** (one number per pixel). For a 64 x 64 image this is 4,096.

**3. Why is flattening needed before Logistic Regression or an SVM?**
These models expect **each sample to be one row of numbers**, so the data forms a 2-D table `X` of shape (number of samples, number of features). A 2-D pixel grid does not fit that shape. Flattening reads the grid row by row and gives a vector. The cost is that the model no longer knows which pixels were neighbours.

**4. Why resize all images to the same size?**
Photos have different sizes, so their flattened vectors would have **different lengths**, and they could not be stacked into one matrix `X`. A model has a fixed number of input features. Resizing guarantees the same length for every image. A smaller size also means fewer features and faster training. The cost is lost fine detail and a possibly stretched shape.

---

## 5. In-lab exercises

**Exercise 1 (bigrams).** With `ngram_range=(1,2)` the vocabulary grows from **11 to 24** features (2.2 times). Every pair of neighbouring words becomes a new feature (for example "the sun", "sun is", "sunny day"), so the vocabulary grows quickly with a larger corpus. In return, some word order is kept.

**Exercise 2 (a 4th document that repeats "sun").** The new document "sun sun sun sun sun and more sun today" adds two new columns (`and`, `more`) and puts **6** in the `sun` column, the biggest number in the matrix. Row sums become 5, 8, 5 and **9**. Raw counts grow with document length and repetition, so a long or repetitive document looks "stronger" even if it is not more relevant. TF-IDF or normalising the counts fixes this.

**Exercise 3 (resize).** The flattened vector has length **H x W** for the original, **4,096** at 64 x 64 and **1,024** at 32 x 32. As the size drops, fine details such as texture, thin lines and small objects disappear first. At 32 x 32 only rough shapes and brightness blobs remain. For my sample photo (a family riding a motorcycle, 333 x 500), the original grayscale image has 166,500 features. At 64 x 64 the overall scene is still clear (the bike, the riders and the basket), but the wheel spokes, faces and the pattern on the bike body are blurred. At 32 x 32 only rough dark and light blobs remain: I can guess where the riders and the wheels are, but small details are gone and the shape is hard to recognise. A small image is cheaper to process, but it carries less information.

**Exercise 4 (`image_to_features`).** The function returns a vector of length **4,096** for three images of different sizes (checked with `assert`). This is why a model can accept all of them.

---

## 6. Home assignment

I built a labelled dataset of **20 images in two classes**: `bike` (10 motorcycle photos) and `car` (10 car photos). The photos come from **Wikimedia Commons** (free-licence images); the file names, licences and source pages are listed in `dataset_sources.csv`. Each image was passed through `image_to_features` with size 64 x 64 and stacked into a matrix.

| Item | Value |
|---|---|
| `X` shape | (20, 4096) |
| `y` shape | (20,) |
| Features per sample (dimensionality) | **4,096** |
| Samples per feature | 20 / 4096 = **0.0049** |
| Saved file | `dataset.npz` |
| Labels | 0 = bike, 1 = car |

**Is 20 samples in 4,096 dimensions a sound basis for learning?**
No. There are about **200 times more features than samples**. This is the "curse of dimensionality". With so many dimensions and so few examples, a model can easily find a rule that separates the 20 training images perfectly by chance (**overfitting**), and then it fails on new images. Distances between points also become less meaningful in very high dimensions. Neighbouring pixels are highly correlated, so many of the 4,096 features carry almost the same information. Better options are: collect far more images (hundreds per class), use a smaller size (for example 16 x 16 = 256 features), reduce dimensions with PCA (Lab 6), or use a CNN (Lab 8), which uses the 2-D structure of the image and shares weights, so it needs far fewer parameters.

---

## 7. Limits of this work

- Bag-of-Words ignores word order and meaning.
- Grayscale throws away colour, and flattening throws away the position of neighbouring pixels.
- 20 images are far too few to train a reliable model. The dataset only demonstrates the transformation.
