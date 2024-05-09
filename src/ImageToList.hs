{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use tuple-section" #-}
module ImageToList (imageToRgb, loadImages, loadImagesNoLabels) where

import Codec.Picture         (readImage, pixelAt, PixelRGB8(..), convertRGB8, Image(..))
import Text.Printf           (printf)
import Control.Monad         (forM)

imageToRgb :: FilePath -> IO [Float]
imageToRgb fp = do
    eimg <- readImage fp
    case eimg of
        Left err -> do
            putStrLn ("Error loading image: " ++ err)
            return []
        Right img -> do
            let img' = convertRGB8 img
            let (Image width height _) = img'
            let pixels = [pixelAt img' x y | x <- [0..width-1], y <- [0..height-1]]
            return $ concatMap (\(PixelRGB8 r g b) -> [fromIntegral r / 255.0, fromIntegral g / 255.0, fromIntegral b / 255.0]) pixels

loadImages :: Int -> FilePath -> IO [([Float], [Float])]
loadImages sizeFolders fp = do
    putStrLn "Loading data..."
    let labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    images <- forM (zip labels [0..]) $ \(label, idx) -> do
        let folder = fp ++ "/" ++ label
        let paths = map (\i -> folder ++ "/" ++ printf "%04d" (i :: Int) ++ ".png") [1..sizeFolders]
        images <- mapM imageToRgb paths
        return $ map (\img -> (oneHot 10 idx, img)) images
    return $ concat images
  where
    oneHot :: Int -> Int -> [Float]
    oneHot numClasses label =
        [ if i == label then 1.0 else 0.0 | i <- [0..numClasses-1] ]

loadImagesNoLabels :: Int -> FilePath -> IO [[Float]]
loadImagesNoLabels sizeFolder fp = do
    putStrLn "Loading data..."
    forM [1..sizeFolder] $ \i -> do
        let path = fp ++ "/" ++ show (i :: Int) ++ ".png"
        imageToRgb path

-- main :: IO ()
-- main = do
--     result <- imageToRgb "app/cifar/data/trainData/airplane/0001.png"
--     print result