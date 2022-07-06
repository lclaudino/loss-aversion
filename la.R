require(MASS)
require(car)
require(emmeans)

df<-read.csv('df.csv',na.strings='NA')

alpha=0.05
res <- rlm(formula=LAMBDA~1 + Task*Dx + Age + IQ + Gender + Shock + MU + STAI_bef + MASQ_GDA + MASQ_GDD + MASQ_AD + MASQ_AA,data=df)
print (Anova(res,3))

# alpha=0.025
print(test(emmeans(res,pairwise~Dx|Task),adjust='sidak',1))

# alpha=0.0125
res2 <- rlm(formula=LAMBDA~1 + Task*Dx*MASQ_AA + MU,data=df)
trends_lambda = emtrends(res2,pairwise ~ Dx, by='Task', var='MASQ_AA')
print(test(trends_lambda),adjust='sidak')


# These are the control analyses (trends over MASQs unspecific to anxiety) whose results populate Figs. 4E and 4F
# alpha = 0.05
res3 <- rlm(formula=LAMBDA~1 + Task*Dx*MASQ_GDA + MU,data=df)
trends_lambda_gda = emtrends(res3,pairwise ~ Dx, by='Task', var='MASQ_GDA')
print(test(trends_lambda_gda))

res4 <- rlm(formula=LAMBDA~1 + Task*Dx*MASQ_AD + MU,data=df)
trends_lambda_ad = emtrends(res4,pairwise ~ Dx, by='Task', var='MASQ_AD')
print(test(trends_lambda_ad))

res5 <- rlm(formula=LAMBDA~1 + Task*Dx*MASQ_GDD + MU,data=df)
trends_lambda_gdd = emtrends(res5,pairwise ~ Dx, by='Task', var='MASQ_GDD')
print(test(trends_lambda_gdd))


# Control analysis that check if any anxiety-related predictor will also be associated with MU:
# alpha = 0.05
res_mu <- rlm(formula=MU~1 + Task*Dx + Age + IQ + Gender + Shock + LAMBDA + STAI_bef + MASQ_GDA + MASQ_GDD + MASQ_AD + MASQ_AA,data=df)
print (Anova(res_mu,3))

