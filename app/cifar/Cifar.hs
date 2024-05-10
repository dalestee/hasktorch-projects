-- Download the data create a data folder and put it inside.
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Unused LANGUAGE pragma" #-}
{-# HLINT ignore "Fuse mapM/map" #-}
{-# HLINT ignore "Missing NOINLINE pragma" #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# HLINT ignore "Use second" #-}
{-# OPTIONS_GHC -Wno-unused-local-binds #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

module Cifar (cifar) where

-- data
import MultClassEvaluation (accuracy, confusionMatrix, confusionMatrix', f1Macro, f1Micro, f1Weighted)
import Func (argmax)

-- image processing
import ImageToList (loadImages)

--hasktorch
import Data.List()
import Torch.Tensor         (Tensor, asTensor, asValue)
import Torch.Functional     (mseLoss, softmax, Dim(..))
import Torch.NN             (flattenParameters, sample)
import Torch.Optim          (foldLoop, mkAdam)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)

import Control.Monad (when)

-- random
import System.Random (newStdGen)
import System.Random.Shuffle (shuffle')

--nlp-tools
import ML.Exp.Chart (drawLearningCurve)

--------------------------------------------------------------------------------
-- Data
--------------------------------------------------------------------------------

randomizeData :: [([Float], [Float])] -> IO [([Float], [Float])]
randomizeData data' = do
    shuffle' data' (length data') <$> newStdGen

--------------------------------------------------------------------------------
-- Training
--------------------------------------------------------------------------------

loss :: MLPParams -> Dim -> (Tensor, Tensor) -> Tensor
loss model dim (input, output) =
    let y = forward model dim input
    in mseLoss y output

forward :: MLPParams -> Dim -> Tensor -> Tensor
forward model dim input = softmax dim output
    where
        output = mlpLayer model input

prediction :: Tensor -> String
prediction output = classes !! argmax (asValue output)
    where
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

validation :: MLPParams ->[(Tensor, Tensor)] -> IO()
validation model dataset = do
    newConfusionMatrix <- confusionMatrix' model dataset
    accuracy' <- accuracy newConfusionMatrix
    f1Micro <- f1Micro newConfusionMatrix
    f1Macro <- f1Macro newConfusionMatrix
    f1Weighted <- f1Weighted newConfusionMatrix
    putStrLn $ "Accuracy: " ++ show accuracy'
    putStrLn $ "F1 Micro: " ++ show f1Micro
    putStrLn $ "F1 Macro: " ++ show f1Macro
    putStrLn $ "F1 Weighted: " ++ show f1Weighted

cifar :: IO ()
cifar = do
    images <- loadImages 5000 "app/cifar/data/trainData"
    rImages <- randomizeData images
    let totalImages = length (take numImages rImages)
    let numTrainingImages = totalImages * 90 `div` 100
    let numValidationImages = totalImages - numTrainingImages
    let (trainingRImages, validationRImages) = splitAt numTrainingImages rImages
    let trainingData = map (\(label, img) -> (asTensor img, asTensor label)) trainingRImages
    let validationData = map (\(label, img) -> (asTensor img, asTensor label)) validationRImages
    putStrLn $ "Training data size: " ++ show (length trainingData)
    -- init <- loadParams hyperParams "app/cifar/models/model-cifar-750_51Acc.pt" -- comment if you want to train from scratch
    init <- sample hyperParams -- comment if you want to load a model
    let opt = mkAdam itr beta1 beta2 (flattenParameters init)
    (trained, _, losses) <- foldLoop (init, opt, []) numEpochs $ \(model, optimizer, losses) i -> do
        let epochLoss = sum (map (loss model dim) trainingData)
        when (i `mod` 1 == 0) $ do
            print ("Epoch: " ++ show i ++ " | Loss: " ++ show (asValue epochLoss :: Float))     
        when (i `mod` 50 == 0) $ do
            print ("Epoch: " ++ show i ++ " | Validation")
            let modelName = "app/cifar/models/model-cifar-256x256" ++ show i ++ ".pt"
            saveParams model modelName
            validation model validationData
        (newState, newOpt) <- update model optimizer epochLoss lr
        return (newState, newOpt, losses :: [Float]) -- without the losses curve
        -- return (newState, newOpt, losses ++ [asValue epochLoss :: Float]) -- with the losses curve

-- --------------------------------------------------------------------------------
-- -- Saving
-- --------------------------------------------------------------------------------

    let maybeLastLoss = if null losses then Nothing else Just (last losses)
    case maybeLastLoss of
        Nothing -> return ()
        Just lastLoss -> do
            let filename = "app/cifar/curves/graph-cifar-mse" ++ show lastLoss ++ ".png"
            drawLearningCurve filename "Learning Curve" [("", losses)]

    let modelName = "app/cifar/models/model-cifar-" ++ -- uncomment if you want to save the model
                     -- uncomment if you want to save the model
                    (if null losses then "noLosses" else show (last losses)) ++ ".pt" -- uncomment if you want to save the model
    -- let modelName = "app/titanic-mlp/models/model-titanic-129.70596_Adam.pt" -- comment if you want to train from scratch
    saveParams trained modelName

--------------------------------------------------------------------------------
-- Validation
--------------------------------------------------------------------------------

    -- imagesTest <- loadImages 1000 "app/cifar/data/testData"
    -- let testData = map (\(label, img) -> (asTensor img, asTensor label)) imagesTest

    -- putStrLn $ "Test data size: " ++ show (length testData)
    -- putStrLn "Evaluating model..."

    -- confusionMatrix <- confusionMatrix "app/cifar/models/model-cifar-600.pt" testData hyperParams

    -- accuracy <- accuracy confusionMatrix

    -- putStrLn $ "Accuracy: " ++ show accuracy

    -- f1Micro <- f1Micro confusionMatrix
    -- f1Macro <- f1Macro confusionMatrix
    -- f1Weighted <- f1Weighted confusionMatrix

    -- putStrLn $ "F1 Micro: " ++ show f1Micro
    -- putStrLn $ "F1 Macro: " ++ show f1Macro
    -- putStrLn $ "F1 Weighted: " ++ show f1Weighted

    -- print confusionMatrix

--------------------------------------------------------------------------------
-- kaggle submission
--------------------------------------------------------------------------------

    -- putStrLn "Predicting test data..."
    -- putStrLn "Loading model..."

    -- model <- loadParams hyperParams "app/cifar/models/model-cifar-750_51Acc.pt"

    -- putStrLn "Loading test data..."

    -- imagesResults <- loadImagesNoLabels 300000 "app/cifar/data/cifar-10/testData/"

    -- putStrLn "Predicting test data..."

    -- -- indexes
    -- -- airplane = 0, automobile = 1, bird = 2, cat = 3, deer = 4, dog = 5, frog = 6, horse = 7, ship = 8, truck = 9
    -- -- write csvfile (id, prediction) ex: (1, cat) if the prediction is 3

    -- let outputs = map (\img -> (img, mlpLayer model (asTensor img))) imagesResults
    -- let outputs' = zipWith (\id (_, output) -> (id, prediction output)) [1..] outputs
    
    -- putStrLn "Writing submission..."
    
    -- writeFile "app/cifar/data/submission.csv" $ unlines $ map (\(img, output) -> show img ++ "," ++ output) outputs'

    -- putStrLn "Done."

    where
        numEpochs = 10000 :: Int
        numImages = 30000 :: Int

        device = Device CPU 0
        -- 32x32 images
        -- ResNet-152
        -- hyperParams = MLPHypParams device 3072 [(2048, Relu), (1024, Relu), (512, Relu), (256, Relu), (128, Relu), (64, Relu), (32, Relu), (16, Relu), (8, Relu), (4, Relu), (2, Relu), (10, Id)] -- very slow to learn can't know if is effective
        hyperParams = MLPHypParams device 3072 [(256, Relu), (256, Relu), (10, Id)] -- not very effective but faster to learn
        -- hyperParams = MLPHypParams device 3072 [(380, Relu), (160, Relu), (80, Relu), (40, Relu), (20, Relu), (10, Id)]

        -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
        itr = 0
        beta1 = 0.9 :: Float
        beta2 = 0.999 :: Float
        lr = 1e-3
        dim = Dim (-1) :: Dim