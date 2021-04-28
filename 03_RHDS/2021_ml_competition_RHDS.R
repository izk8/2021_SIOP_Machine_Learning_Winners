####Team RHDS ML Competition Script####

rm(list = ls())

library(tidyverse)


setwd("siop_ml_competition\\")#set this to wherever you have the data saved
train <- read_csv("train.csv")
participant_dev <- read_csv("participant_dev.csv")
participant_test <- read_csv("participant_test.csv")



overall_dat<-
  bind_rows(
    train,
    participant_dev,
    participant_test
  )


##check predictor averages by projected group status
overall_dat%>%
  group_by(Protected_Group)%>%
  summarise_at(
    vars(
      starts_with("PScale"),
      contains("Time")
    ),
    list(
      ~mean(.,na.rm=T)
    )
  )%>%
  View()



####data preprocess
overall_dat1<-
  overall_dat%>%
  
  ##categorize SJ_MOST and SJ_LEAST
  mutate_at(
    vars(
      starts_with("SJ_Most"),
      starts_with("SJ_Least"),
      starts_with("Biodata")
    ),
    ~as.character(.)
  )%>%
  
  ##remove where protected class unavailable
  filter(
    !(is.na(Protected_Group)&split=="train")
  )%>%
  
  ##replace NA with "MISSING" for categorical fields
  mutate_at(
    vars(
      which(sapply(.,class)=="character"),
      -split
    ),
    list(
      ~ifelse(
        is.na(.),
        "MISSING",
        .
      )
    )
  )




####set model parameters
##predictors
predictors<-
  names(
    overall_dat1%>%
      select(
        starts_with("SJ_Most"),
        starts_with("SJ_Least"),
        starts_with("Scenario"),
        starts_with("Biodata"),
        starts_with("PScale"),
        contains("Time")
      )
  )

##data
dat<-overall_dat1

##ml options
use_pca<-"no"
use_smote<-"no"
train_prop<-0.95

##outcome
outcome<-c("Retained","High_Performer")

##building model paramters
model_parameters<-
  list(
    predictors=predictors,
    dat=dat,
    use_pca=use_pca,
    use_smote=use_smote,
    train_prop=train_prop,
    outcome=outcome
  )




####function to preprocess data and output model results####
####script can be run all together or in chunks
preprocess_model_train<-
  function(model_parameters){
    
    
    
    ####create folders dynamically based on date
    ##get date
    date_cleaned<-
      gsub(
        "\\-",
        "_",
        Sys.Date()
      )
    
    create_output_folders<-
      function(x){
        if(
          !dir.exists(path=paste0(getwd(),"/",x,"/",date_cleaned,"_1"))
        ){
          
          dir.create(paste0(getwd(),"/",x,"/",date_cleaned,"_1"))
          
        }else{
          
          dir.create(
            paste0(
              getwd(),"/",x,"/",date_cleaned,"_",
              max(as.numeric(
                sapply(
                  str_split(
                    sapply(
                      str_split(
                        grep(
                          date_cleaned,
                          list.dirs(path=paste(getwd(),x,sep="/")),
                          value=TRUE
                        ),
                        pattern="/"
                      ),
                      "[",8
                    ),
                    pattern="_"
                  ),
                  "[",4
                )
              ))+1
            )
          )
        }
      }
    output_locations<-c("model_output")
    lapply(output_locations,create_output_folders)
    
    ##set date of interest dynamically
    date_of_interest<-
      paste0(
        date_cleaned,"_",
        max(as.numeric(
          sapply(
            str_split(
              sapply(
                str_split(
                  grep(
                    date_cleaned,
                    list.dirs(path=paste(getwd(),"model_output",sep="/")),
                    value=TRUE
                  ),
                  pattern="/"
                ),
                "[",8
              ),
              pattern="_"
            ),
            "[",4
          )
        ))
      )
    
    
    
    
    predictors<-model_parameters$predictors
    dat<-model_parameters$dat
    use_pca<-model_parameters$use_pca
    outcome<-model_parameters$outcome
    use_smote<-model_parameters$use_smote
    
    dat1<-dat%>%
      select(predictors)
    
    
    ####data cleaning/preprocessing
    ##Dummy coding categorical fields
    dummy_dat<-
      sjmisc::to_dummy(
        dat1%>%
          select(
            which(sapply(.,class)=="character")
          ),
        suffix="label"
      )
    
    
    ####Removing fields with near zero variance
    ##packages for preprocessing/ML tasks
    library(caret)
    library(RANN)
    ##Get a list of all near zero variance fields
    near_zero_var_fields<-
      nearZeroVar(
        dummy_dat,
        freqCut=95/5,
        names=TRUE
      )
    
    ##number of removed fields
    print(length(near_zero_var_fields))
    print(ncol(dummy_dat))
    
    ##get differences between included fields
    included_categorical_fields<-
      tibble(fields=names(dummy_dat))%>%
      left_join(
        tibble(
          fields=near_zero_var_fields,
          merge_check=1
        ),
        by="fields"
      )%>%
      filter(
        is.na(merge_check)
      )%>%
      pull(fields)
    print(included_categorical_fields)
    
    ##removing fields with near zero variance
    dummy_dat1<-
      dummy_dat%>%
      select(
        -near_zero_var_fields
      )
    
    ##Pre process to remove highly correlated fields
    categorical_dat_prep<-
      preProcess(
        dummy_dat1, 
        method=c("corr")
      )
    #remove the fields
    dummy_dat2<-
      predict(
        categorical_dat_prep,
        dummy_dat1
      )
    print(ncol(dummy_dat2))
    
    ##get differences between included fields
    remove_intercorrelated_fields<-
      tibble(fields=names(dummy_dat1))%>%
      left_join(
        tibble(
          fields=names(dummy_dat2),
          merge_check=1
        ),
        by="fields"
      )%>%
      filter(
        is.na(merge_check)
      )%>%
      pull(fields)
    print(remove_intercorrelated_fields)
    
    
    ####Numeric field cleaning
    ##Get all numeric fields except for outcome and ID
    numeric_fields<-
      dat1%>%
      select(
        which(sapply(.,class)=="integer"),
        which(sapply(.,class)=="numeric")
      )
    
    ##get near zero variance fields
    near_zero_var_fields_numeric<-
      nearZeroVar(
        numeric_fields,
        names=TRUE
      )
    print(near_zero_var_fields_numeric)
    
    ##remove the fields
    numeric_fields1<-
      numeric_fields%>%
      select(
        -near_zero_var_fields_numeric
      )
    
    
    ####using categorical data for numeric missing data imputation
    ##bind dummy coded cleaned categorical fields
    all_dat<-
      as.data.frame(
        bind_cols(
          dummy_dat2,
          numeric_fields1
        )
      )
    
    ##setting preprocess for missing data impute
    all_field_preprocess<-
      preProcess(
        all_dat, 
        method=
          c(
            "center",
            "scale"
          )
      )
    
    ##impute the data
    all_dat1<-
      predict(
        all_field_preprocess,
        all_dat
      )
    
    ##regression imputation for missing data
    library(mice)
    all_dat1_impute<-mice(all_dat1,method = "norm.nob",m=1)
    all_dat2<-complete(all_dat1_impute)
    
    
    ####transform numeric data
    ##get numeric fields
    numeric_fields2<-
      all_dat2%>%
      select(
        names(numeric_fields1)
      )
    
    ##if we want to use PCA
    if(use_pca=="yes"){
      ##set preprocess function
      numeric_field_preprocess<-
        preProcess(
          numeric_fields2,
          method="pca"
        )
      
      ##perform transformation
      numeric_fields2<-
        predict(
          numeric_field_preprocess,
          numeric_fields2
        )
    }
    
    
    ####get final data set by binding ID, outcome, dummy coded data (need to do this because dummy values centered and scaled and we need these to be 0 and 1), and transformed numeric data
    model_train_dat<-
      bind_cols(
        dummy_dat2,
        numeric_fields2,
        dat%>%
          select(
            UNIQUE_ID,
            Protected_Group,
            Retained,
            High_Performer,
            split
          )
      )
    
    predictors1<-
      c(
        names(dummy_dat2),
        names(numeric_fields2)
      )
    
    
    ####function to train models for each outcome
    train_models<-
      function(outcome,model_train_dat,predictors1,date_of_interest,use_smote){
        
        ##get all data with outcome for training model
        model_train_dat1<-
          model_train_dat%>%
          select(
            predictors1,
            Protected_Group,
            outcome,
            UNIQUE_ID
          )%>%
          filter_at(
            vars(outcome),
            all_vars(!is.na(.))
          )
        
        ##get all data without outcome to generate predictions on later
        model_train_dat1a<-
          model_train_dat%>%
          select(
            predictors1,
            Protected_Group,
            outcome,
            UNIQUE_ID
          )%>%
          filter_at(
            vars(outcome),
            all_vars(is.na(.))
          )
        
        
        ####Splitting sample into train, test, and validation
        ##Set for replicability
        set.seed(4444)
        train_prop<-train_prop
        
        ##Creating data partitions based on outcome
        #training data
        train_index<-
          createDataPartition(
            model_train_dat1%>%pull(outcome),
            p=train_prop
          )$Resample1
        train<-
          model_train_dat1[train_index,]
        train1<-
          train%>%
          mutate(
            MODEL_GROUP=paste("train",sep="_")
          )
        
        if(
          use_smote=="yes"&
          outcome=="High_Performer"
        ){
          
          train2<-
            train%>%
            rename_at(
              vars(all_of(outcome)),
              list(
                ~gsub(outcome,"output",.)
              )
            )%>%
            mutate_at(
              vars(output),
              list(
                ~as.factor(.)
              )
            )%>%
            select(
              -UNIQUE_ID,
              -Protected_Group
            )
          
          train_resampled<-
            DMwR::SMOTE(
              output~.,
              data=train2
            )
          train3<-
            train_resampled%>%
            mutate(
              output=as.numeric(output)-1
            )%>%
            rename_at(
              vars(output),
              list(
                ~gsub("output",outcome,.)
              )
            )
          
          train<-train3
        }
        
        #Test/validation data (Everything not included in training)
        test<-
          model_train_dat1[-train_index,]
        test1<-
          test%>%
          mutate(
            MODEL_GROUP=paste("test",sep="_")
          )
        
        
        ####building the model
        ##load xgboost
        library(xgboost)
        library(tictoc)
        ##random grid search for hp tuning
        hp_grid<-data.frame()
        #number of iterations
        for(iter in 1:50){
          
          {tic()
            print(iter)
            #parameter grid
            param<-
              list(
                objective="binary:logistic",
                eval_metric="auc",
                max_depth=sample(5:10,1),
                eta=runif(1,.01,.3),
                gamma=runif(1,0.0,0.2),
                subsample=runif(1,.6,.9),
                colsample_bytree=runif(1,.5,.8),
                min_child_weight=sample(1:40,1)
              )
            #Other model parameters
            cv.nround<-sample(seq(1000,10000,by=500),1)
            cv.nfold<-sample(5:10,1)
            nthread<-sample(5:10,1)
            
            ##get data for model
            #predictors
            model_dat<-
              as.matrix(
                train%>%
                  select(
                    predictors1
                  )
              )
            #outcome
            outcome1<-train%>%pull(outcome)
            
            ##building the model to get parameters
            mdcv<-
              xgb.cv(
                data=model_dat,
                label=outcome1,
                params=param, 
                nthread=nthread,
                nfold=cv.nfold, 
                nrounds=cv.nround,
                verbose=FALSE, #
                early_stopping_rounds=5
              )
            
            ##getting hyperparameters based on best auc
            max_auc<-max(mdcv$evaluation_log[,test_auc_mean])
            max_auc_index<-which.max(mdcv$evaluation_log[,test_auc_mean])
            mdcv$best_iteration
            
            ##get the hyperparamters
            hyper_parameters<-
              mdcv$evaluation_log%>%
              filter(iter==mdcv$best_iteration)%>%
              select(-iter)%>%
              bind_cols(
                as_tibble(param),
                max_auc_index=max_auc_index,
                cv.nround=cv.nround,
                cv.nfold=cv.nfold,
                nthread=nthread
              )
            
            ##get the final output
            hp_grid<-
              bind_rows(
                hp_grid,
                hyper_parameters
              )
            toc()}
        }
        
        ##add column that ranks hp quality
        hp_grid1<-
          hp_grid%>%
          arrange(desc(max_auc_index))%>%
          mutate(
            hp_rank=1:n()
          )
        
        ##writing grid to file
        write_csv(
          hp_grid1,
          paste0(
            "model_output/",
            date_of_interest,
            "/",
            outcome,
            "_HP_GRID.csv"
          )
        )
        
        
        ####create predictions based on multiple models
        number_of_models<-5
        
        ##output models as files
        for(iter in 1:number_of_models){
          #get paramaters
          hyper_parameters<-
            hp_grid1%>%
            filter(
              hp_rank==iter
            )
          print(iter)
          print(hyper_parameters)
          #specify parameters
          param<-
            list(
              objective=
                hyper_parameters$objective,
              eval_metric=
                hyper_parameters$eval_metric,
              max_depth=
                hyper_parameters$max_depth,
              eta=
                hyper_parameters$eta,
              gamma=
                hyper_parameters$gamma,
              subsample=
                hyper_parameters$subsample,
              colsample_bytree=
                hyper_parameters$colsample_bytree,
              min_child_weight=
                hyper_parameters$min_child_weight
            )
          
          ##build the model
          xgbm<-
            xgboost(
              params=param,
              data=
                as.matrix(
                  train%>%
                    select(
                      predictors1
                    )
                ),
              label=train%>%pull(outcome),
              nrounds=hyper_parameters$max_auc_index,
              nthread=hyper_parameters$nthread
            )
          
          ##output the model
          xgb.save(
            xgbm,
            paste0(
              "model_output/",
              date_of_interest,
              "/",
              outcome,
              "_XGB_MODEL_",
              iter,
              ".model"
            )
          )
        }
        
        ##Get the importance matrix for best model
        importance_matrix<-
          xgb.importance(
            feature_names=
              names(
                train%>%
                  select(
                    predictors1
                  )
              ),
            
            model=
              xgb.load(
                paste0(
                  "model_output/",
                  date_of_interest,
                  "/",
                  outcome,
                  "_XGB_MODEL_1.model"
                )
              )
            
          )
        
        #output the matrix
        write_csv(
          importance_matrix,
          paste0(
            "model_output/",
            date_of_interest,
            "/",
            outcome,
            "_IMPORTANCE_MATRIX.xlsx"
          )
        )
        
        
        ####load models and generate predictions
        xgb_get_preds<-
          function(x,dat){
            
            ##load the model
            xgbm<-
              xgb.load(
                paste0(
                  "model_output/",
                  date_of_interest,
                  "/",
                  outcome,
                  "_XGB_MODEL_",
                  x,
                  ".model"
                )
              )
            
            ##get the predicted values
            preds<-
              predict(
                xgbm,
                as.matrix(
                  dat%>%
                    select(
                      predictors1
                    )
                )
              )
            
            preds<-
              tibble(preds)%>%
              rename_all(
                list(
                  ~paste(.,x,sep="")
                )
              )
            return(preds)
            
          }
        
        ##apply function
        #set number of models
        number_of_models<-5
        #get predictions
        predicted_values<-
          rowMeans(
            bind_cols(
              lapply(
                1:number_of_models,
                xgb_get_preds,
                bind_rows(
                  train1,
                  test,
                  model_train_dat1a
                )
              )
            )
          )
        
        
        ####evaluation metrics and data output
        ##get predictions and group info together
        output_dat<-
          bind_rows(
            train1,
            test1,
            model_train_dat1a
          )%>%
          select(
            -predictors1
          )%>%
          mutate(
            PREDS=predicted_values,
            PREDS1=
              ifelse(
                PREDS>=0.5,1,0
              )
          )
        
        
        ####set function to do evaluations at different data subsets
        filters_used<-
          tibble(
            train_all="MODEL_GROUP=='train'",
            train_protected="MODEL_GROUP=='train'&Protected_Group==1",
            train_non_protected="MODEL_GROUP=='train'&Protected_Group==0",
            test_all="MODEL_GROUP=='test'",
            test_protected="MODEL_GROUP=='test'&Protected_Group==1",
            test_non_protected="MODEL_GROUP=='test'&Protected_Group==0"
          )
        
        eval_calc<-
          function(filters,filters_used,output_dat,outcome){
            
            filter_to_use<-
              filters_used%>%
              pull(
                filters
              )
            
            output_dat2<-
              output_dat%>%
              filter(
                eval(
                  parse(
                    text=filter_to_use
                  )
                )
              )
            
            roc_calc<-
              pROC::roc(
                output_dat2%>%pull(outcome),
                output_dat2$PREDS1
              )
            
            eval_outcome<-
              tibble(
                eval_group=filters,
                sensitivity=roc_calc$sensitivities[2],
                specificity=roc_calc$specificities[2],
                auc=as.numeric(roc_calc$auc)
              )
            
            return(eval_outcome)
          }
        
        
        eval_metrics<-
          bind_rows(
            lapply(
              names(filters_used),
              eval_calc,
              filters_used,
              output_dat,
              outcome
            )
          )
        
        write_csv(
          eval_metrics,
          paste0(
            "model_output/",
            date_of_interest,
            "/eval_metrics_",
            outcome,
            ".csv"
          )
        )
        
        output_dat1<-
          output_dat%>%
          select(
            UNIQUE_ID,
            MODEL_GROUP,
            PREDS
          )%>%
          rename_at(
            vars(
              MODEL_GROUP,
              PREDS
            ),
            list(
              ~paste(.,outcome,sep="_")
            )
          )
        
        return(output_dat1)
        
      }
    
    
    
    #full_evaluation_data<-
    #  left_join(
    #    output_dat1_retained,
    #    output_dat1_higher_performer,
    #    by="UNIQUE_ID"
    #  )%>%
    #  right_join(
    #    model_train_dat%>%
    #      select(
    #        UNIQUE_ID,
    #        Protected_Group,
    #        Retained,
    #        High_Performer,
    #        split
    #      ),
    #    by="UNIQUE_ID"
    #  )
    
    
    full_evaluation_data<-
      lapply(
        outcome,
        train_models,
        model_train_dat,
        predictors1,
        date_of_interest,
        use_smote
      )%>%
      purrr::reduce(full_join,by="UNIQUE_ID")%>%
      right_join(
        model_train_dat%>%
          select(
            UNIQUE_ID,
            Protected_Group,
            Retained,
            High_Performer,
            split
          ),
        by="UNIQUE_ID"
      )
    
    
    write_csv(
      full_evaluation_data,
      paste0(
        "model_output/",
        date_of_interest,
        "/full_eval_data.csv"
      )
    )
    
    
    full_evaluation_data<-
      read_csv(
        paste0(
          "model_output/",
          date_of_interest,
          "/full_eval_data.csv"
        )
      )
    
    ####accuracy metric calculation for optimal predictor weighting####
    weights<-seq(0,1,.01)
    
    weight_optimize<-
      function(weights,dat){
        
        
        number_to_retain<-
          round(
            nrow(dat)*0.5,
            digits = 0
          )
        
        dat1<-
          dat%>%
          mutate(
            app_score=
              (PREDS_Retained*weights)+(PREDS_High_Performer*(1-weights))
          )%>%
          arrange(
            desc(app_score)
          )%>%
          mutate(
            app_ranking=1:n(),
            Hire=
              ifelse(app_ranking<=number_to_retain,1,0)
          )
        
        
        ####get output metrics
        ##Percentage_of_true_top_performers_hired = The ratio of top performers you hired (based on your predictions) out of the possible top hires you could have hired. Selected top performers divided by top performers in the data set. This constitutes 25% of the overall accuracy score.
        top_performers_in_dat<-
          ifelse(
            sum(dat1$High_Performer)>number_to_retain,
            number_to_retain,
            sum(dat1$High_Performer)
          )
        top_performers_hired<-
          nrow(dat1%>%filter(Hire==1&High_Performer==1))
        Percentage_of_true_top_performers_hired<-
          top_performers_hired/top_performers_in_dat
        
        ##Percentage_of_true_retained_hired = The ratio of employees you hired (based on your predictions) that stayed with the company out of the possible retained hires you could have made. Selected retained employees divided by retained employees possible. This constitutes 25% of the overall accuracy score.
        true_retained_in_dat<-
          ifelse(
            sum(dat1$Retained)>number_to_retain,
            number_to_retain,
            sum(dat1$Retained)
          )
        true_retained_hired<-
          nrow(dat1%>%filter(Hire==1&Retained==1))
        Percentage_of_true_retained_hired<-
          true_retained_hired/true_retained_in_dat
        
        ##Percentage_of_true_retained_top_performers_hired = The ratio of employees you hired (based on your predictions) that stayed with the company and were top performers out of the possible top performing retained hires. Selected top performing retained employees divided by top performing retained employees in the data set. This constitutes 50% of the overall accuracy score.
        top_performers_true_retained_in_dat<-
          ifelse(
            nrow(
              dat1%>%
                filter(
                  High_Performer==1&
                    Retained==1
                )
            )>number_to_retain,
            number_to_retain,
            nrow(
              dat1%>%
                filter(
                  High_Performer==1&
                    Retained==1
                )
            )
          )
        top_performers_true_retained_hired<-
          nrow(dat1%>%filter(Hire==1&High_Performer&Retained==1))
        Percentage_of_true_retained_top_performers_hired<-
          top_performers_true_retained_hired/top_performers_true_retained_in_dat
        
        
        ##Overall_accuracy
        Overall_accuracy<-
          Percentage_of_true_top_performers_hired*25+
          Percentage_of_true_retained_hired*25+
          Percentage_of_true_retained_top_performers_hired*50
        
        
        ##Adverse_impact_ratio = The protected group selection ratio (protected hired over the protected in the applicant pool) divided by the non-protected selection ratio (non-protected hired over the non-protected in the applicant pool). Selection_rate_of_protected divided by Selection_rate_of_non_protected.
        Selection_rate_of_protected<-
          nrow(
            dat1%>%
              filter(
                Protected_Group==1&
                  Hire==1
              )
          )/
          nrow(
            dat1%>%
              filter(
                Protected_Group==1
              )
          )
        Selection_rate_of_non_protected<-
          nrow(
            dat1%>%
              filter(
                Protected_Group==0&
                  Hire==1
              )
          )/
          nrow(
            dat1%>%
              filter(
                Protected_Group==0
              )
          )
        Adverse_impact_ratio<-
          Selection_rate_of_protected/Selection_rate_of_non_protected
        
        
        ##Unfairness
        Unfairness<-
          abs(1-Adverse_impact_ratio)*100
        
        
        ##Final_score
        Final_score<-
          Overall_accuracy-Unfairness
        
        
        output_metrics<-
          tibble(
            
            PREDS_Retained_weights=
              weights,
            
            PREDS_High_Performer_weights=
              1-weights,
            
            top_performers_in_dat=
              top_performers_in_dat,
            
            top_performers_hired=
              top_performers_hired,
            
            Percentage_of_true_top_performers_hired=
              Percentage_of_true_top_performers_hired,
            
            true_retained_in_dat=
              true_retained_in_dat,
            
            true_retained_hired=
              true_retained_hired,
            
            Percentage_of_true_retained_hired=
              Percentage_of_true_retained_hired,
            
            top_performers_true_retained_in_dat=
              top_performers_true_retained_in_dat,
            
            top_performers_true_retained_hired=
              top_performers_true_retained_hired,
            
            Percentage_of_true_retained_top_performers_hired=
              Percentage_of_true_retained_top_performers_hired,
            
            Overall_accuracy=
              Overall_accuracy,
            
            Selection_rate_of_protected=
              Selection_rate_of_protected,
            
            Selection_rate_of_non_protected=
              Selection_rate_of_non_protected,
            
            Adverse_impact_ratio=
              Adverse_impact_ratio,
            
            Unfairness=
              Unfairness,
            
            Final_score=
              Final_score
          )
        
        
        return(output_metrics)
      }
    
    
    accuracy_optimize_all<-
      bind_rows(
        lapply(
          weights,
          weight_optimize,
          full_evaluation_data%>%
            filter(
              split=="train"&
                !is.na(High_Performer)&
                !is.na(Retained)
            )
        )
      )%>%
      arrange(
        desc(Final_score)
      )
    
    
    write_csv(
      accuracy_optimize_all,
      paste0(
        "model_output/",
        date_of_interest,
        "/accuracy_weight_optimization.csv"
      )
    )
    
    
    accuracy_optimize_test_only<-
      bind_rows(
        lapply(
          weights,
          weight_optimize,
          full_evaluation_data%>%
            filter(
              MODEL_GROUP_Retained=="test"&
                MODEL_GROUP_High_Performer=="test"
            )
        )
      )%>%
      arrange(
        desc(Final_score)
      )
    
    
    write_csv(
      accuracy_optimize_test_only,
      paste0(
        "model_output/",
        date_of_interest,
        "/accuracy_weight_optimization_test_data_only.csv"
      )
    )
    
    
    accuracy_optimize_train_only<-
      bind_rows(
        lapply(
          weights,
          weight_optimize,
          full_evaluation_data%>%
            filter(
              MODEL_GROUP_Retained=="train"&
                MODEL_GROUP_High_Performer=="train"
            )
        )
      )%>%
      arrange(
        desc(Final_score)
      )
    
    
    write_csv(
      accuracy_optimize_train_only,
      paste0(
        "model_output/",
        date_of_interest,
        "/accuracy_weight_optimization_train_data_only.csv"
      )
    )
    
    
    ####generate hires of dev and test data using optimized weights
    ##get the weights
    number_of_weights_to_average<-1
    
    #accuracy_optimize_all
    #accuracy_optimize_train_only
    #accuracy_optimize_test_only
    accuracy_optimize1<-
      accuracy_optimize_all%>%
      mutate(
        n=1:n()
      )%>%
      filter(
        n<=number_of_weights_to_average
      )
    
    PREDS_Retained_weights<-
      mean(accuracy_optimize1$PREDS_Retained_weights)
    
    PREDS_High_Performer_weights<-
      mean(accuracy_optimize1$PREDS_High_Performer_weights)
    
    
    
    ####use weights to create data for submission
    ##dev
    dev_hired<-
      round(
        0.5*
          nrow(
            full_evaluation_data%>%
              filter(
                split=="dev"
              )
          ),
        digits = 0
      )
    dev_output<-
      full_evaluation_data%>%
      filter(
        split=="dev"
      )%>%
      mutate(
        app_score=
          (
            (PREDS_Retained_weights*PREDS_Retained)+
              (PREDS_High_Performer_weights*PREDS_High_Performer)
          )
      )%>%
      arrange(
        desc(app_score)
      )%>%
      mutate(
        app_ranking=1:n(),
        Hire=
          ifelse(app_ranking<=dev_hired,1,0)
      )%>%
      select(
        UNIQUE_ID,
        Hire
      )
    
    write_csv(
      dev_output,
      paste0(
        "model_output/",
        date_of_interest,
        "/dev_output_all_dat_4_weights_",
        date_of_interest,
        ".csv"
      )
    )
    
    ##test
    test_hired<-
      round(
        0.5*
          nrow(
            full_evaluation_data%>%
              filter(
                split=="test"
              )
          ),
        digits = 0
      )
    test_output<-
      full_evaluation_data%>%
      filter(
        split=="test"
      )%>%
      mutate(
        app_score=
          (
            (PREDS_Retained_weights*PREDS_Retained)+
              (PREDS_High_Performer_weights*PREDS_High_Performer)
          )
      )%>%
      arrange(
        desc(app_score)
      )%>%
      mutate(
        app_ranking=1:n(),
        Hire=
          ifelse(app_ranking<=test_hired,1,0)
      )%>%
      select(
        UNIQUE_ID,
        Hire
      )
    
    write_csv(
      test_output,
      paste0(
        "model_output/",
        date_of_interest,
        "/test_output_all_dat_4_weights_",
        date_of_interest,
        ".csv"
      )
    )
    
    model_parameter_output<-
      bind_cols(
        model_no=date_of_interest,
        predictors=model_parameters$predictors,
        use_pca=model_parameters$use_pca,
        use_smote=model_parameters$use_smote,
        train_percent=model_parameters$train_prop
      )
    
    write_csv(
      model_parameter_output,
      paste0(
        "model_output/",
        date_of_interest,
        "/model_parameters_",
        date_of_interest,
        ".csv"
      )
    )
    
  }



lapply(
  list(
    model_parameters
    ),
  preprocess_model_train
)









