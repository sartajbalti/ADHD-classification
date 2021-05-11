flod5data is our data;i have already segement it into 5-flod for the 5-fold-crossvalidation
HK_helm_train_1h.m is the code for khelm only one layer. i have write 4 scrips for one layer to four layer .
you can add more layer to record the result
i help you rewrite the file main2_k.m and main1_hk_1h.m. The first one is the experient for keml, and the
 second one is for hkelm with one layer.
you can through this two file to learn how to modify parameter.
And you have to record some results that are the classification accuracy of every flod:
for main2_k.m ,NC_Fold and ADHD_Fold are used to record result;
for main1_hk_1h.m,AD_result and NC_result are used to record result;