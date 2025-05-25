# Aspect-Based Sentiment Analysis for Restaurant Reviews in Estonian
This repository contains code and datasets for performing aspect-based sentiment analysis (ABSA) on restaurant reviews written in Estonian as part of a term paper.
## Repository Structure

* metrics_function.ipynb — contains custom metrics function that calculates precision, recall, f1 and accuracy in different modes (only aspect- or the whole triplet-level), as well as postiprocessing functions

* absa_estonian_restaurant_reviews_dataset — directory related to the first open-source manually annotated Estonian restaurant review dataset

* unsupervised_rule_based_baseline — directory related to the unsupervised rule-based baseline model

* llm_experiments — directory related LLM experiments performed with the resources of cHARISMa (unfortunately, Gemma3 code was not run and thus not included in the term paper because of incompatible library versions and wrong config.json. Each subdirectory inside has a name, for example: _annot_0_adj_eng_ means that task instruction type is annotation guidelines, number of in-context examples = 0, prompt is adjusted to model errors, language of the prompt is English.

## Credits
Research supervisor — Eduard Klyshinsky, Associate Professor at the School of Linguistics

Research advisor — Anna Aksenova, NLP Researcher
