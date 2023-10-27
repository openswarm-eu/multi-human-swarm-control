# Comment code: Ctrl + Shift + C
# Run entire script: Ctrl + Shift + Enter

# Load library
library(dplyr)

################
# Points scored
################

# Load csv
pointsScored <- read.csv(file = 'points_scored.csv')

# Display data
print(pointsScored)

# Normality test
pointsScored.DIR <- pointsScored %>% filter(Communication == 'DIR')
pointsScored.IND <- pointsScored %>% filter(Communication == 'IND')
pointsScored.RS <- pointsScored %>% filter(Condition == 'RS')
pointsScored.NRS <- pointsScored %>% filter(Condition == 'NRS')

pointsScored.DIR.RS <- pointsScored.DIR %>% filter(Condition == 'RS')
pointsScored.IND.RS <- pointsScored.IND %>% filter(Condition == 'RS')
pointsScored.DIR.NRS <- pointsScored.DIR %>% filter(Condition == 'NRS')
pointsScored.IND.NRS <- pointsScored.IND %>% filter(Condition == 'NRS')

mean(pointsScored.DIR$Points)
sd(pointsScored.DIR$Points)
mean(pointsScored.IND$Points)
sd(pointsScored.IND$Points)

mean(pointsScored.DIR.RS$Points)
sd(pointsScored.DIR.RS$Points)
mean(pointsScored.DIR.NRS$Points)
sd(pointsScored.DIR.NRS$Points)
mean(pointsScored.IND.RS$Points)
sd(pointsScored.IND.RS$Points)
mean(pointsScored.IND.NRS$Points)
sd(pointsScored.IND.NRS$Points)

shapiro.test(pointsScored.DIR$Points)
shapiro.test(pointsScored.IND$Points)
shapiro.test(pointsScored.RS$Points)
shapiro.test(pointsScored.NRS$Points)

# Mean
aggregate(pointsScored$Points, list(pointsScored$Communication), FUN=mean) 
aggregate(pointsScored$Points, list(pointsScored$Condition), FUN=mean) 

# Statistical significance
t.test(Points ~ Communication, pointsScored, alternative = 'two.sided')
t.test(Points ~ Condition, pointsScored, alternative = 'two.sided', paired=TRUE)
wilcox.test(Points ~ Communication, data=pointsScored, alternative = 'two.sided', exact=FALSE)
wilcox.test(Points ~ Condition, data=pointsScored, alternative = 'two.sided', paired=TRUE)
wilcox.test(pointsScored.RS$Points, pointsScored.NRS$Points, alternative = 'two.sided', paired=TRUE)

wilcox.test(pointsScored.DIR.RS$Points, pointsScored.DIR.NRS$Points, alternative = 'two.sided', paired=TRUE, exact=FALSE)
wilcox.test(pointsScored.DIR.RS$Points, pointsScored.IND.RS$Points, alternative = 'two.sided', exact=FALSE)
wilcox.test(pointsScored.DIR.NRS$Points, pointsScored.IND.NRS$Points, alternative = 'two.sided', exact=FALSE)
wilcox.test(pointsScored.IND.RS$Points, pointsScored.IND.NRS$Points, alternative = 'two.sided', paired=TRUE, exact=FALSE)

# wilcox.test(pointsScored.DIR.RS$Points, pointsScored.IND$Points, alternative = 'two.sided', exact=FALSE)

####################
# Distance traveled
####################

# Load csv
traveledDistance <- read.csv(file = 'distance_traveled.csv')

# Display data
print(traveledDistance)

traveledDistance.DIR <- traveledDistance %>% filter(Communication == 'DIR')
traveledDistance.IND <- traveledDistance %>% filter(Communication == 'IND')
traveledDistance.RS <- traveledDistance %>% filter(Condition == 'RS')
traveledDistance.NRS <- traveledDistance %>% filter(Condition == 'NRS')

traveledDistance.DIR.RS <- traveledDistance.DIR %>% filter(Condition == 'RS')
traveledDistance.IND.RS <- traveledDistance.IND %>% filter(Condition == 'RS')
traveledDistance.DIR.NRS <- traveledDistance.DIR %>% filter(Condition == 'NRS')
traveledDistance.IND.NRS <- traveledDistance.IND %>% filter(Condition == 'NRS')

mean(traveledDistance.RS$Distance)
sd(traveledDistance.RS$Distance)
mean(traveledDistance.NRS$Distance)
sd(traveledDistance.NRS$Distance)

mean(traveledDistance.DIR.RS$Distance)
sd(traveledDistance.DIR.RS$Distance)
mean(traveledDistance.DIR.NRS$Distance)
sd(traveledDistance.DIR.NRS$Distance)
mean(traveledDistance.IND.RS$Distance)
sd(traveledDistance.IND.RS$Distance)
mean(traveledDistance.IND.NRS$Distance)
sd(traveledDistance.IND.NRS$Distance)

# Normality test
shapiro.test(traveledDistance.DIR$Distance)
shapiro.test(traveledDistance.IND$Distance)
shapiro.test(traveledDistance.RS$Distance)
shapiro.test(traveledDistance.NRS$Distance)

# Statistical significance
# t.test(Distance ~ Communication, data=traveledDistance)
t.test(Distance ~ Condition, data=traveledDistance)
wilcox.test(Distance ~ Communication, data=traveledDistance, alternative = 'two.sided', exact=FALSE)
wilcox.test(Distance ~ Condition, data=traveledDistance, alternative = 'two.sided', paired=TRUE)
wilcox.test(Points ~ Communication, data=pointsScored, alternative = 'two.sided', exact=FALSE)
wilcox.test(Points ~ Condition, data=pointsScored, alternative = 'two.sided', paired=TRUE, exact=FALSE)

################
# Team distance
################

# Load csv
teamDistance <- read.csv(file = 'team_distance.csv')

# Display data
print(teamDistance)

teamDistance.DIR <- teamDistance %>% filter(Communication == 'DIR')
teamDistance.IND <- teamDistance %>% filter(Communication == 'IND')
teamDistance.RS <- teamDistance %>% filter(Condition == 'RS')
teamDistance.NRS <- teamDistance %>% filter(Condition == 'NRS')

teamDistance.DIR.RS <- teamDistance.DIR %>% filter(Condition == 'RS')
teamDistance.IND.RS <- teamDistance.IND %>% filter(Condition == 'RS')
teamDistance.DIR.NRS <- teamDistance.DIR %>% filter(Condition == 'NRS')
teamDistance.IND.NRS <- teamDistance.IND %>% filter(Condition == 'NRS')

mean(teamDistance.RS$Distance)
sd(teamDistance.RS$Distance)
mean(teamDistance.NRS$Distance)
sd(teamDistance.NRS$Distance)

mean(teamDistance.DIR.RS$Distance)
sd(teamDistance.DIR.RS$Distance)
mean(teamDistance.DIR.NRS$Distance)
sd(teamDistance.DIR.NRS$Distance)
mean(teamDistance.IND.RS$Distance)
sd(teamDistance.IND.RS$Distance)
mean(teamDistance.IND.NRS$Distance)
sd(teamDistance.IND.NRS$Distance)

# Normality test
shapiro.test(teamDistance.DIR$Distance)
shapiro.test(teamDistance.IND$Distance)
shapiro.test(teamDistance.RS$Distance)
shapiro.test(teamDistance.NRS$Distance)

# Statistical significance
# t.test(Distance ~ Communication, data=teamDistance)
# t.test(Distance ~ Send, data=teamDistance)
wilcox.test(Distance ~ Communication, data=teamDistance, alternative = 'two.sided', exact=FALSE)
wilcox.test(Distance ~ Condition, data=teamDistance, alternative = 'two.sided', paired=TRUE, exact=FALSE)
wilcox.test(Points ~ Communication, data=pointsScored, alternative = 'two.sided', exact=FALSE)
wilcox.test(Points ~ Condition, data=pointsScored, alternative = 'two.sided', paired=TRUE, exact=FALSE)

################
# Robots shared
################

# Load csv
robotShared <- read.csv(file = 'robot_shared.csv')

# Display data
print(robotShared)

robotShared.DIR.RS <- robotShared %>% filter(Communication == 'DIR-RS')
robotShared.IND.RS <- robotShared %>% filter(Communication == 'IND-RS')

mean(robotShared.DIR.RS$Robots.Sent)
sd(robotShared.DIR.RS$Robots.Sent)
mean(robotShared.IND.RS$Robots.Sent)
sd(robotShared.IND.RS$Robots.Sent)

mean(robotShared.DIR.RS$Robots.Sent)
sd(robotShared.DIR.RS$Robots.Sent)
mean(robotShared.IND.RS$Robots.Sent)
sd(robotShared.IND.RS$Robots.Sent)

# Normality test
shapiro.test(robotShared.DIR.RS$Robots.Sent)
shapiro.test(robotShared.IND.RS$Robots.Sent)

# Statistical significance
wilcox.test(Robots.Sent ~ Communication, data=robotShared, alternative='two.sided', exact=FALSE)

################
# Traveler time
################

# Load csv
travelerTime <- read.csv(file = 'traveler_time.csv')

# Display data
print(travelerTime)

travelerTime.DIR.RS <- travelerTime %>% filter(Communication == 'DIR-RS')
travelerTime.IND.RS <- travelerTime %>% filter(Communication == 'IND-RS')

# Normality test
shapiro.test(travelerTime.DIR.RS$Time)
shapiro.test(travelerTime.IND.RS$Time)

# Statistical significance
t.test(Time ~ Communication, data=travelerTime, alternative='less', exact=FALSE)
wilcox.test(Time ~ Communication, data=travelerTime, alternative='less', exact=FALSE)

####################
# Traveler distance
####################

# Load csv
travelerDist <- read.csv(file = 'traveler_dist.csv')

# Display data
print(travelerDist)

travelerDist.DIR.RS <- travelerDist %>% filter(Communication == 'DIR-RS')
travelerDist.IND.RS <- travelerDist %>% filter(Communication == 'IND-RS')

# Normality test
shapiro.test(travelerDist.DIR.RS$Distance)
shapiro.test(travelerDist.IND.RS$Distance)

# Statistical significance
t.test(Distance ~ Communication, data=travelerDist, alternative='two.sided', exact=FALSE)
wilcox.test(Distance ~ Communication, data=travelerDist, alternative='two.sided', exact=FALSE)

############################
# Average traveler distance
############################

# Load csv
averageTravelerDist <- read.csv(file = 'average_traveler_dist.csv')

# Display data
print(averageTravelerDist)

averageTravelerDist.DIR.RS <- averageTravelerDist %>% filter(Communication == 'DIR-RS')
averageTravelerDist.IND.RS <- averageTravelerDist %>% filter(Communication == 'IND-RS')

# Normality test
shapiro.test(averageTravelerDist.DIR.RS$Distance)
shapiro.test(averageTravelerDist.IND.RS$Distance)

# Statistical significance
t.test(Distance ~ Communication, data=averageTravelerDist, alternative='two.sided', exact=FALSE)
wilcox.test(Distance ~ Communication, data=averageTravelerDist, alternative='two.sided', exact=FALSE)

##################
# Global taskload
##################

# Load csv
globalTaskload <- read.csv(file = 'global_taskload.csv')

# Display data
print(globalTaskload)

globalTaskload.DIR <- globalTaskload %>% filter(Communication == 'DIR')
globalTaskload.IND <- globalTaskload %>% filter(Communication == 'IND')
globalTaskload.RS <- globalTaskload %>% filter(Condition == 'RS')
globalTaskload.NRS <- globalTaskload %>% filter(Condition == 'NRS')

globalTaskload.DIR.RS <- globalTaskload.DIR %>% filter(Condition == 'RS')
globalTaskload.IND.RS <- globalTaskload.IND %>% filter(Condition == 'RS')
globalTaskload.DIR.NRS <- globalTaskload.DIR %>% filter(Condition == 'NRS')
globalTaskload.IND.NRS <- globalTaskload.IND %>% filter(Condition == 'NRS')

mean(na.omit(globalTaskload.RS$Average.Taskload))
sd(na.omit(globalTaskload.RS$Average.Taskload))
mean(na.omit(globalTaskload.NRS$Average.Taskload))
sd(na.omit(globalTaskload.NRS$Average.Taskload))

mean(globalTaskload.DIR.RS$Average.Taskload)
sd(globalTaskload.DIR.RS$Average.Taskload)
length(na.omit(globalTaskload.DIR.RS$Average.Taskload))
mean(na.omit(globalTaskload.DIR.RS$Average.Taskload))
sd(na.omit(globalTaskload.DIR.RS$Average.Taskload))

mean(globalTaskload.DIR.NRS$Average.Taskload)
sd(globalTaskload.DIR.NRS$Average.Taskload)
mean(globalTaskload.IND.RS$Average.Taskload)
sd(globalTaskload.IND.RS$Average.Taskload)
mean(globalTaskload.IND.NRS$Average.Taskload)
sd(globalTaskload.IND.NRS$Average.Taskload)


# Normality test
shapiro.test(globalTaskload.DIR$Average.Taskload)
shapiro.test(globalTaskload.IND$Average.Taskload)
shapiro.test(globalTaskload.RS$Average.Taskload)
shapiro.test(globalTaskload.NRS$Average.Taskload)

# F-test
var.test(globalTaskload.RS$Average.Taskload, globalTaskload.NRS$Average.Taskload, alternative = 'two.sided')

# Statistical significance
t.test(Average.Taskload ~ Communication, data=globalTaskload, alternative = 'two.sided')
# t.test(Average.Taskload ~ Condition, data=globalTaskload, paired=TRUE, alternative = 'two.sided')
t.test(globalTaskload.RS$Average.Taskload, globalTaskload.NRS$Average.Taskload, paired = TRUE, alternative = 'two.sided')

# wilcox.test(Average.Taskload ~ Communication, data=globalTaskload)
wilcox.test(Average.Taskload ~ Condition, data=globalTaskload, alternative = 'two.sided', paired = TRUE, exact = FALSE)
wilcox.test(globalTaskload.RS$Average.Taskload, globalTaskload.NRS$Average.Taskload, paired = TRUE, alternative = 'two.sided')

##############################
# Global situational awareness
##############################

# Load csv
globalSA <- read.csv(file = 'global_situational_awareness.csv')

# Display data
print(globalSA)

globalSA.DIR <- globalSA %>% filter(Communication == 'DIR')
globalSA.IND <- globalSA %>% filter(Communication == 'IND')
globalSA.RS <- globalSA %>% filter(Condition == 'RS')
globalSA.NRS <- globalSA %>% filter(Condition == 'NRS')

globalSA.DIR.RS <- globalSA.DIR %>% filter(Condition == 'RS')
globalSA.IND.RS <- globalSA.IND %>% filter(Condition == 'RS')
globalSA.DIR.NRS <- globalSA.DIR %>% filter(Condition == 'NRS')
globalSA.IND.NRS <- globalSA.IND %>% filter(Condition == 'NRS')

mean(na.omit(globalSA.RS$Average.Situational.Awareness))
sd(na.omit(globalSA.RS$Average.Situational.Awareness))
mean(na.omit(globalSA.NRS$Average.Situational.Awareness))
sd(na.omit(globalSA.NRS$Average.Situational.Awareness))

mean(globalSA.DIR.RS$Average.Situational.Awareness)
sd(globalSA.DIR.RS$Average.Situational.Awareness)
mean(globalSA.DIR.NRS$Average.Situational.Awareness)
sd(globalSA.DIR.NRS$Average.Situational.Awareness)
mean(globalSA.IND.RS$Average.Situational.Awareness)
sd(globalSA.IND.RS$Average.Situational.Awareness)
length(na.omit(globalSA.IND.RS$Average.Situational.Awareness))
mean(na.omit(globalSA.IND.RS$Average.Situational.Awareness))
sd(na.omit(globalSA.IND.RS$Average.Situational.Awareness))
mean(globalSA.IND.NRS$Average.Situational.Awareness)
sd(globalSA.IND.NRS$Average.Situational.Awareness)

wilcox.test(globalSA.DIR$Average.Situational.Awareness, globalSA.IND$Average.Situational.Awareness, alternative = 'two.sided')
wilcox.test(globalSA.RS$Average.Situational.Awareness, globalSA.NRS$Average.Situational.Awareness, paired = TRUE, alternative = 'two.sided')

