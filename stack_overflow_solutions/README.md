 
#### Description from Kaggle 
Millions of programmers use Stack Overflow to get high quality answers to their programming questions every day.  We take quality very seriously, and have evolved an effective culture of moderation to safe-guard it.

With more than six thousand new questions asked on Stack Overflow every weekday we're looking to add more sophisticated software solutions to our moderation toolbox.

Closing Questions

Currently about 6% of all new questions end up "closed".  Questions can be closed as off topic, not constructive, not a real question, or too localized.  More in depth descriptions of each reason can be found in the Stack Overflow FAQ.  The exact duplicate close reason has been excluded from this contest, since it depends on previous questions.

Your goal is to build a classifier that predicts whether or not a question will be closed given the question as submitted, along with the reason that the question was closed.  Additional data about the user at question creation time is also available.

Data containes following columns: 
###### Input
-PostCreationDate
-OwnerUserId
-OwnerCreationDate
-ReputationAtPostCreation
-OwnerUndeletedAnswerCountAtPostTime
-Title
-BodyMarkdown
-Tag1
-Tag2
-Tag3
-Tag4
-Tag5
###### Output
-OpenStatus

For this project I use text contained in collumns "Title" and "BodyMarkdown"

-  src_fasttext: Here I use fasttext embedding vectors. I clean the text and use the embeddings for each word in a sentence and average them togther to get the sentences embeddings.  
