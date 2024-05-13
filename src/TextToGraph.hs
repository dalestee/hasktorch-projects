module TextToGraph (textToGraph) where

import ML.Exp.Chart (drawLearningCurve)

import Data.List.Split (splitOn)
import Data.List (isInfixOf)

textFileToList :: FilePath -> IO [Float]
textFileToList filePath = do
    content <- readFile filePath
    let lines' = lines content
    let filteredLines = skipValidationLines lines'
    let losses = map (read . drop 6 . head . tail . splitOn "|") filteredLines
    return losses

skipValidationLines :: [String] -> [String]
skipValidationLines [] = []
skipValidationLines (x:xs)
    | "Validation" `isInfixOf` x = skipValidationLines (drop 4 xs)
    | otherwise = x : skipValidationLines xs

makeGraph :: [Float] -> IO ()
makeGraph data' = do
    drawLearningCurve "app/cifar/curves/lossCifar.png" "Learning Curve" [("", data')]

textToGraph :: FilePath -> IO ()
textToGraph filePath = do
    data' <- textFileToList filePath
    makeGraph data'

main :: IO ()
main = do
    textToGraph "app/cifar/losses.txt"
    putStrLn "Done"