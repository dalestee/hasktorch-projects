{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Unused LANGUAGE pragma" #-}
{-# HLINT ignore "Fuse mapM/map" #-}
{-# HLINT ignore "Missing NOINLINE pragma" #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}

module Titanic (titanic) where

-- data
import Evaluation (accuracy, precision, recall, extractBinary)

--hasktorch
import Control.Monad (when)
import Data.List()
import Torch.Tensor         (Tensor, asTensor, asValue)
import Torch.Functional     (mseLoss)
import Torch.NN             (flattenParameters)
import Torch.Optim          (foldLoop, mkAdam)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)

-- other libraries for parsing
import System.IO.Unsafe (unsafePerformIO)
import Data.List.Split (splitOn)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools

--------------------------------------------------------------------------------
-- Data
--------------------------------------------------------------------------------

-- Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
parseData :: String -> IO [([Float], Float)]
parseData filePath = do
    content <- readFile filePath
    let lines' = tail $ lines content
    return $ map (\line -> let fields = map read (splitOn "," line) :: [Float] in
        (tail fields, head fields)) lines'

splitData :: [([Float], Float)] -> ([([Float], Float)], [([Float], Float)])
splitData dataset = 
    let trainSize = round ((0.9 :: Double )* fromIntegral (length dataset)) :: Int
    in splitAt trainSize dataset

-- Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
parseDataTest :: [([Float], Int)]
parseDataTest = unsafePerformIO $ do
    content <- readFile "app/titanic-mlp/data/test-mod.csv"
    let lines' = tail $ lines content
    return $ map (\line -> let fields = splitOn "," line in
        ([read (fields !! 1), read (fields !! 2), read (fields !! 3), read (fields !! 4), read (fields !! 5), read (fields !! 6), read (fields !! 7)], read (fields !! 0))) lines'

loss :: MLPParams -> ([Float], Float) -> Tensor
loss model (input, output) = let y = mlpLayer model (asTensor input)
                             in mseLoss y (asTensor output)

arrayToCSV :: [(Int, Int)] -> String
arrayToCSV = unlines . map (\(a, b) -> show a ++ "," ++ show b)

titanic :: IO ()
titanic = do

    fullTrainData <- parseData "app/titanic-mlp/data/train-mod.csv"
    let (trainingData, validationData) = splitData fullTrainData
    testData <- parseData "app/titanic-mlp/data/test-mod.csv"

    -- init <- sample hyperParams -- commented out because we are loading a pre-trained model
    init <- loadParams hyperParams "app/titanic-mlp/models/model-titanic-129.70596_Adam.pt" -- comment if you want to train from scratch
    let opt = mkAdam itr beta1 beta2 (flattenParameters init)
    -- let opt = GD
    (trained, _, losses, valLosses) <- foldLoop (init, opt, [], []) numIters $ \(model, optimizer, losses, valLosses) i -> do
        let epochLoss = sum (map (loss model) trainingData)
        let valLoss = sum (map (loss model) validationData)
        when (i `mod` 1 == 0) $ do
            putStrLn $ "Epoch: " ++ show i
            putStrLn $ "Epoch Loss: " ++ show epochLoss
            putStrLn $ "Validation Loss: " ++ show valLoss
        (newState, newOpt) <- update model optimizer epochLoss lr
        return (newState, newOpt, losses ++ [asValue epochLoss :: Float], valLosses ++ [asValue valLoss :: Float])

    case losses of
        [] -> return ()
        _  -> do
            let lastLoss = last losses
            let filename = "app/titanic-mlp/curves/graph-titanic-mse" ++ show lastLoss ++ ".png"
            drawLearningCurve filename "Learning Curve" [("", losses)]

    -- let modelName = "app/titanic-mlp/models/model-titanic-" ++ -- uncomment if you want to save the model
    --                 (if null losses then "noLosses" else show (last losses)) ++ ".pt" -- uncomment if you want to save the model
    let modelName = "app/titanic-mlp/models/model-titanic-129.70596_Adam.pt" -- comment if you want to train from scratch
    saveParams trained modelName

    model <- loadParams hyperParams modelName


    -- validation

    let (tp, tn, fp, fn) = extractBinary validationData model
    putStrLn $ "True Positives: " ++ show tp
    putStrLn $ "True Negatives: " ++ show tn
    putStrLn $ "False Positives: " ++ show fp
    putStrLn $ "False Negatives: " ++ show fn
    putStrLn $ "Accuracy: " ++ show (accuracy tp tn (tp + tn + fp + fn))
    putStrLn $ "Precision: " ++ show (precision tp fp)
    putStrLn $ "Recall: " ++ show (recall tp fn)

    -- test data
    let testData = parseDataTest
        outputs = map (\(input, passengerId) -> (passengerId, mlpLayer model (asTensor input))) testData
        outputs' = map (\(passengerId, output) -> (passengerId, if (asValue output :: Float) > 0.5 then 1 else 0)) outputs
    writeFile "app/titanic-mlp/data/submission.csv" $ arrayToCSV outputs'
    return ()

    where
        numIters = 10
        device = Device CPU 0
        hyperParams = MLPHypParams device 7 [(21, Relu), (1, Id)]
        -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
        itr = 0
        beta1 = 0.9
        beta2 = 0.999
        lr = 1e-2

--------------------------------------------------------------------------------
-- Validation
--------------------------------------------------------------------------------

