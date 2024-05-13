{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use tuple-section" #-}
module ImageToList (imageToRgb, loadImages, loadImagesNoLabels) where

import Codec.Picture         (readImage, pixelAt, PixelRGB8(..), convertRGB8, Image(..), DynamicImage, DynamicImage(ImageRGB8))
import Text.Printf           (printf)
import Control.Monad         (forM)

imageToRgb :: DynamicImage -> [Float]
imageToRgb dynamicImage =
    case dynamicImage of
        ImageRGB8 image -> imageToRgbList image
        _ -> error "Unsupported image format"

imageToRgbList :: Image PixelRGB8 -> [Float]
imageToRgbList image = do
    let width = imageWidth image
    let height = imageHeight image
    let pixels = [ pixelAt image x y | y <- [0..height-1], x <- [0..width-1] ]
    let rgb = map (\(PixelRGB8 r g b) -> [fromIntegral r, fromIntegral g, fromIntegral b]) pixels
    concat rgb


loadImages :: Int -> FilePath -> IO [([Float], [Float])]
loadImages sizeFolders fp = do
    putStrLn "Loading data..."
    let labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    images <- forM (zip labels [0..]) $ \(label, idx) -> do
        let folder = fp ++ "/" ++ label
        let paths = map (\i -> folder ++ "/" ++ printf "%04d" (i :: Int) ++ ".png") [1..sizeFolders]
        images <- forM paths $ \path -> do
            image <- readImage path
            either (error . show) (return . imageToRgb) image
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
        either (error . show) imageToRgb <$> readImage path

-- main :: IO ()
-- main = do
--     result <- either (error . show) imageToRgb <$> readImage "app/cifar/data/trainData/airplane/0001.png"
--     print result