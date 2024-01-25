# Search the Given Data with LLM

Train a Model to give suggestions for a Product search with the given dataset.
Dataset:
Data Source- This Data source has more than 20L product names for an e-commerce search
auto-complete.

## Data Cleaning:

• It is Identified that Column ‘Product Name’ has 6k Duplicates and it was removed
before processing the Data.
• Convert all the product names to lower case for consistency.
• Numbers are not removed from the Names since it has dependency on the
specifications of the Product.
• Remove ‘\n’ and ‘\r’ from the Product names.
• Remove Commas and Hyphens
• Remove stop words and get the root words of the existing words by stemming.
Runtime: Colab Python3 T4 GPU
Approach:
Model embedding-based search approach was used instead of Model training. Since the
whole task is similarity matching, Model embeddings should be the best approach for this
scenario.
Logical Approach:

Search input query in the pandas DF. It is not an expected approach but, this approach
should be considered because it consumes less time and memory compared to model based
approach.

## Model based Approach:
• Instead of training the model, pre-trained language models were loaded, and the
training data was converted to vector embeddings.
• Embeddings were stored in a .h5 file for later usage.
• For the inference, both the embeddings and training data were loaded and used to
map the identified index from the model in training Data.
Cosine similarity is used to identify the similar embeddings from the training data with the
search query embedding.
Models Experimented:
1. Paraphrase-MiniLM-L6-v2
2. BERT

Model used: Paraphrase-MiniLM-L6-v2
This is a sentence transformer model. Since this model is specifically trained for
paraphrasing it can easily understand the semantic similarity between the sentences. Since it
is a mini model it will be lightweight and give fast inferences compared to other large
models.
Limitations:
- Due to Infra dependency only 4L names are used for embeddings.
- Specific issue is with the Memory leak in Tokenizer
- Embeddings needs to be generated after each data update.
Why BERT is not used,
BERT is not performing good with the minimum words like ‘Fire TV’. But it is performing
good with more words like ‘Kindle Oasis E-reader with Leather Charging Cover - Merlot, 6
High-Resolution Display (300 ppi), Wi-Fi - Includes Special Offers’.
To fix the issue, Padding and Truncation were tried but no luck.
BERT was trained on whole Data.
Note: BERT doesn’t have much infra dependency
How to make it better:
- Optimize the Data cleansing process
- Look for other approaches in Tokenizing with full GPU offloading
- Compare the cosine similarity with other similarity algorithms
Test cases:
1. "E-reader, 6 High-Resolution Display"

['Kindle Paperwhite E-reader - White, 6 High-Resolution Display (300 ppi) with Built-
in Light, Wi-Fi - Includes Special Offers,,','Kindle Voyage E-reader, 6 High-Resolution

Display (300 ppi) with Adaptive Built-in Light, PagePress Sensors, Wi-Fi - Includes
Special Offers,','Kindle Voyage E-reader, 6 High-Resolution Display (300 ppi) with
Adaptive Built-in Light, PagePress Sensors, Wi-Fi - Includes Special Offers','All-New
Kindle Oasis E-reader - 7 High-Resolution Display (300 ppi), Waterproof, Built-In
Audible, 32 GB, Wi-Fi - Includes Special Offers','All-New Kindle Oasis E-reader - 7
High-Resolution Display (300 ppi), Waterproof, Built-In Audible, 8 GB, Wi-Fi - Includes

Special Offers','Kindle Oasis E-reader with Leather Charging Cover - Merlot, 6 High-
Resolution Display (300 ppi), Wi-Fi - Includes Special Offers','Kindle Oasis E-reader

with Leather Charging Cover - Merlot, 6 High-Resolution Display (300 ppi), Wi-Fi -
Includes Special Offers,,','Kindle Oasis E-reader with Leather Charging Cover - Walnut,

6 High-Resolution Display (300 ppi), Wi-Fi - Includes Special Offers','All-New Kindle
Oasis E-reader - 7 High-Resolution Display (300 ppi), Waterproof, Built-In Audible, 32
GB, Wi-Fi + Free Cellular Connectivity','Kindle Voyage E-reader, 6 High-Resolution
Display (300 ppi) with Adaptive Built-in Light, PagePress Sensors, Free 3G + Wi-Fi -
Includes Special Offers']
2. "Certified Refurbished Amazon Echo"
['Certified Refurbished Amazon Echo','Certified Refurbished Amazon Fire TV with
Alexa Voice Remote','Certified Refurbished Amazon Fire TV Stick (Previous
Generation - 1st),,,\r\nKindle Paperwhite,,,','Certified Refurbished Amazon Fire TV
(Previous Generation - 1st),,,\r\nCertified Refurbished Amazon Fire TV (Previous
Generation - 1st),,,','Certified Refurbished Amazon Fire TV with Alexa Voice
Remote,,,\r\nCertified Refurbished Amazon Fire TV with Alexa Voice
Remote,,,','Certified Refurbished Amazon Fire TV Stick (Previous Generation -
1st),,,\r\nCertified Refurbished Amazon Fire TV Stick (Previous Generation -
1st),,,','Apple iPhone 6 Plus Silver 16GB Unlocked Smartphone (Certified
Refurbished)','Motorola V551 Refurbishd Cell Phone Unlocked','Apple iPhone 5 16GB
- Unlocked - White (Certified Refurbished)','Apple iPhone 6 Plus Gold 64GB Unlocked
Smartphone (Certified Refurbished)']
3. "Tablet with Alexa"
['Amazon Fire HD 8 with Alexa (8" HD Display Tablet)','Fire Tablet with Alexa, 7
Display, 16 GB, Magenta - with Special Offers','Fire Tablet with Alexa, 7" Display, 16
GB, Magenta - with Special Offers','Fire HD 8 Tablet with Alexa, 8 HD Display, 16 GB,
Tangerine - with Special Offers,','Fire HD 8 Tablet with Alexa, 8 HD Display, 16 GB,
Tangerine - with Special Offers','Fire Tablet with Alexa, 7 Display, 16 GB, Blue - with
Special Offers','Fire HD 8 Tablet with Alexa, 8 HD Display, 32 GB, Tangerine - with
Special Offers,','Fire HD 8 Tablet with Alexa, 8 HD Display, 32 GB, Tangerine - with
Special Offers','Fire HD 8 Tablet with Alexa, 8" HD Display, 32 GB, Tangerine - with
Special Offers','All-New Fire HD 8 Tablet with Alexa, 8 HD Display, 16 GB, Marine Blue
- with Special Offers']
