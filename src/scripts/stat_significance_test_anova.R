# Script adapted from
# https://www.datanovia.com/en/lessons/mixed-anova-in-r/#two-way-mixed

library(tidyverse)
library(ggpubr)
library(rstatix)

# Wide format
# set.seed(123)
my_data = read.csv(file='points_scored.csv')
# my_data = read.csv(file='distance_traveled.csv')
# my_data = read.csv(file='robot_shared.csv')
# my_data = read.csv(file='team_distance.csv')
# my_data = read.csv(file='global_taskload.csv')
# my_data = read.csv(file='global_situational_awareness.csv')


# anxiety %>% sample_n_by(group, size = 1)

# # Gather the columns t1, t2 and t3 into long format.
# # Convert id and time into factor variables
# anxiety <- anxiety %>%
#   gather(key = "time", value = "score", t1, t2, t3) %>%
#   convert_as_factor(id, time)
# # Inspect some random rows of the data by groups
# set.seed(123)
# anxiety %>% sample_n_by(group, time, size = 1)

# Summary statistics
my_data %>%
  group_by(Condition, Communication) %>%
  get_summary_stats(Points, type = "mean_sd")

# Visualization
bxp <- ggboxplot(
  my_data, x = "Communication", y = "Points",
  color = "Condition", palette = "jco"
)
bxp

# Check assumptions

# Outliers
my_data %>%
  group_by(Condition, Communication) %>%
  identify_outliers(Points)

# Normality assumption
my_data %>%
  group_by(Condition, Communication) %>%
  shapiro_test(Points)

ggqqplot(my_data, "Points", ggtheme = theme_bw()) +
  facet_grid(Condition ~ Communication)

# Homogneity of variance assumption
my_data %>%
  group_by(Condition) %>%
  levene_test(Points ~ Communication)

# Homogeneity of covariances assumption
box_m(my_data[, "Points", drop = FALSE], my_data$Communication)

# Two-way mixed ANOVA test
res.aov <- anova_test(
  data = my_data, dv = Points, wid = Users,
  between = Communication, within = Condition
)
get_anova_table(res.aov)

# Post-hoc tests

# Procedure for non-significant two-way interaction

# Within-subject
my_data %>%
  pairwise_wilcox_test(
    Points ~ Condition, paired = TRUE, 
    p.adjust.method = "bonferroni"
  )
# Between-subject
my_data %>%
  pairwise_wilcox_test(
    Points ~ Communication, 
    p.adjust.method = "bonferroni"
  )
