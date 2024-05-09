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
import MultClassEvaluation (accuracy, precision, recall, extractBinaryMultClassification)
import Func (argmax)

-- image processing
import ImageToList (loadImages, loadImagesNoLabels)

--hasktorch
import Data.List()
import Torch.Tensor         (Tensor, asTensor, asValue)
import Torch.Functional     (mseLoss, softmax, Dim(..))
import Torch.NN             (flattenParameters, sample)
import Torch.Optim          (foldLoop, mkAdam)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)

-- random
import System.Random (newStdGen)
import System.Random.Shuffle (shuffle')

-- other libraries for parsing
-- import System.IO.Unsafe (unsafePerformIO)
-- import Data.List.Split (splitOn)

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
-- dim is the number of classes to be predicted
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

cifar :: IO ()
cifar = do
    -- images <- loadImages "app/cifar/data/trainData"
    -- -- print type of image
    -- images <- randomizeData images
    -- print images
    -- images <- loadImages 5000 "app/cifar/data/trainData"
    -- rImages <- randomizeData images
    -- let trainingData = take numImages $ map (\(label, img) -> (asTensor img, asTensor label)) rImages
--     putStrLn $ "Training data size: " ++ show (length trainingData)
--     init <- sample hyperParams
--     let opt = mkAdam itr beta1 beta2 (flattenParameters init)
--     (trained, _, losses) <- foldLoop (init, opt, []) numEpochs $ \(model, optimizer, losses) i -> do
--         let epochLoss = sum (map (loss model dim) trainingData)
--         when (i `mod` 1 == 0) $ do
--             print i
--             print epochLoss
--         when (i `mod` 50 == 0) $ do
--             let modelName = "app/cifar/models/model-cifar-" ++ show i ++ ".pt"
--             saveParams model modelName
--         (newState, newOpt) <- update model optimizer epochLoss lr
--         return (newState, newOpt, losses :: [Float]) -- without the losses curve
--         -- return (newState, newOpt, losses ++ [asValue epochLoss :: Float]) -- with the losses curve

-- --------------------------------------------------------------------------------
-- -- Saving
-- --------------------------------------------------------------------------------

--     let maybeLastLoss = if null losses then Nothing else Just (last losses)
--     case maybeLastLoss of
--         Nothing -> return ()
--         Just lastLoss -> do
--             let filename = "app/cifar/curves/graph-cifar-mse" ++ show lastLoss ++ ".png"
--             drawLearningCurve filename "Learning Curve" [("", losses)]

--     let modelName = "app/cifar/models/model-cifar-" ++ -- uncomment if you want to save the model
--                      -- uncomment if you want to save the model
--                     (if null losses then "noLosses" else show (last losses)) ++ ".pt" -- uncomment if you want to save the model
--     -- let modelName = "app/titanic-mlp/models/model-titanic-129.70596_Adam.pt" -- comment if you want to train from scratch
--     saveParams trained modelName

--------------------------------------------------------------------------------
-- Validation
--------------------------------------------------------------------------------

    imagesTest <- loadImages 1000 "app/cifar/data/testData"
    let testData = map (\(label, img) -> (asTensor img, asTensor label)) imagesTest

    putStrLn $ "Test data size: " ++ show (length testData)
    putStrLn "Evaluating model..."

    (tp, fp, fn) <- extractBinaryMultClassification "app/cifar/models/model-cifar-750_51Acc.pt" testData hyperParams

    putStrLn $ "accuracy: " ++ show (accuracy tp fp fn)
    putStrLn $ "precision: " ++ show (precision tp fp)
    putStrLn $ "recall: " ++ show (recall tp fn)

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
        numImages = 1000 :: Int
        device = Device CPU 0
        -- 32x32 images
        -- ResNet-152
        -- hyperParams = MLPHypParams device 3072 [(2048, Relu), (1024, Relu), (512, Relu), (256, Relu), (128, Relu), (64, Relu), (32, Relu), (16, Relu), (8, Relu), (4, Relu), (2, Relu), (10, Id)] -- very slow to learn can't know if is effective
        hyperParams = MLPHypParams device 3072 [(256, Relu), (256, Relu), (10, Id)] -- not very effective but faster to learn
        -- hyperParams = MLPHypParams device 3072 [(380, Relu), (160, Relu), (80, Relu), (40, Relu), (20, Relu), (10, Id)]
        -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
        itr = 0 :: Integer
        beta1 = 0.9 :: Float
        beta2 = 0.999 :: Float
        lr = 1e-3 :: Float
        dim = Dim (-1) :: Dim