**About Problem Statement**:
 - Genre: NLP
 - Problem Type: Contextual Semantic Similarity, Auto generate Text-based answers

**Submission Format**:
 - You need to generate upto 3 distractors for each Question-Answer combination
 - Each distractor is a string
 - The 3 distractors/strings need to be separated with a comma (,)
 - Each value in Results.csv's distractor column will contain the distractors as follows:
	distractor_for_QnA_1 = "distractor1","distractor2","distractor3"

**About the Evaluation Parameter**:
 - All distractor values for 1 question-answer will be converted into a vector form
 - 1 vector gets generated for submitted distractors and 1 vector is generated for truth value
 - cosine_similarity between these 2 vectors is evaluated
 - Similarly, cosine_similarity gets evaluated for all the question-answer combinations
 - Score of your submitted prediction file = mean ( cosine_similarity between distractor vectors for each entry in test.csv)

 
**My approach?**:

 - **Pre-processing of dataset**: Pre-processing mainly involves splitting all the columns based on punctuation, while preserving all the blank-spaces as we're gonna need them while constructing distractors.
 - **Construction of distractors**: Objective was to change results for questions in such a way that they can be appropriate distractors. So I have targetted specific type of words for changing them by their antonyms and synonyms. 
 - **Methodology used for changing words** : I've used a method that creates distractors in 7 stages. In first stages it finds if the sentence has a number, if yes it increments it for creating other distractors. If no number is there it finds propositions, if not found it finds adjectives, next is conjunctions, adpositions, nouns and adverbs. Why this specific order of search? Well it was giving best set of results, I tried different permutations of these linguistic entities. 
 - **How to find antonym and synonyms?** : I've used wordnet from NLTK, for finding antonyms and synonyms. As silky(hacker earth support) said I can use word embeddings for the code. 

 **THANK YOU!!**
