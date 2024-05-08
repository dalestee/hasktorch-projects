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

module Cifar (cifar) where

-- image processing
import ImageToList (loadImages)

-- data
-- import Evaluation (accuracy, precision, recall, extractBinary)

--hasktorch
import Control.Monad (when)
import Data.List()
import Torch.Tensor         (Tensor, asTensor, asValue, shape)
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

randomizeData :: [(Int, [Float])] -> IO [(Int, [Float])]
randomizeData data' = do
    shuffle' data' (length data') <$> newStdGen

loss :: MLPParams -> Dim -> (Tensor, Tensor) -> Tensor
-- dim is the number of classes to be predicted
loss model dim (input, output) = 
    let y = forward model dim input
    in mseLoss y output

forward :: MLPParams -> Dim -> Tensor -> Tensor
forward model dim input = softmax dim $ mlpLayer model input

cifar :: IO ()
cifar = do
    -- images <- loadImages "app/cifar/data/trainData"
    -- -- print type of image
    -- images <- randomizeData images
    -- print images
    images <- loadImages "app/cifar/data/trainData"
    rImages <- randomizeData images
    let trainingData = take numImages $ map (\(label, img) -> (asTensor img, asTensor ([fromIntegral label] :: [Float]))) rImages    -- print tensor size
    putStrLn $ "Training data size: " ++ show (length trainingData)
    init <- sample hyperParams
    let opt = mkAdam itr beta1 beta2 (flattenParameters init)
    (trained, _, losses) <- foldLoop (init, opt, []) numEpochs $ \(model, optimizer, losses) i -> do
        let epochLoss = sum (map (loss model dim) trainingData)
        when (i `mod` 1 == 0) $ do
            print i
            print epochLoss
        (newState, newOpt) <- update model optimizer epochLoss lr
        return (newState, newOpt, losses :: [Float]) -- without the losses curve
        -- return (newState, newOpt, losses ++ [asValue epochLoss :: Float]) -- with the losses curve

    -- rest of your code

    let maybeLastLoss = if null losses then Nothing else Just (last losses)
    case maybeLastLoss of
        Nothing -> return ()
        Just lastLoss -> do
            let filename = "app/titanic-mlp/curves/graph-titanic-mse" ++ show lastLoss ++ ".png"
            drawLearningCurve filename "Learning Curve" [("", losses)]

    let modelName = "app/titanic-mlp/models/model-titanic-" ++ -- uncomment if you want to save the model
                     -- uncomment if you want to save the model
                    (if null losses then "noLosses" else show (last losses)) ++ ".pt" -- uncomment if you want to save the model
    -- let modelName = "app/titanic-mlp/models/model-titanic-129.70596_Adam.pt" -- comment if you want to train from scratch
    saveParams trained modelName

    where
        numEpochs = 100
        numImages = 5000
        device = Device CPU 0
        -- 32x32 images
        -- ResNet-152
        hyperParams = MLPHypParams device 3072 [(2048, Relu), (1024, Relu), (512, Relu), (256, Relu), (128, Relu), (64, Relu), (32, Relu), (16, Relu), (8, Relu), (4, Relu), (2, Relu), (10, Id)]
        -- hyperParams = MLPHypParams device 3072 [(256, Relu), (256, Relu), (10, Id)]
        -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
        itr = 0
        beta1 = 0.9
        beta2 = 0.999
        lr = 1e-1
        dim = Dim 0



