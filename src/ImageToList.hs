module ImageToList (imageToRgb, loadImages) where

import Codec.Picture         (readImage, pixelAt, PixelRGB8(..), DynamicImage, convertRGB8, Image(..))
import System.FilePath.Posix (splitExtension)
import Data.Word             (Word8)
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
            let (filename, ext) = splitExtension fp
            let img' = convertRGB8 img
            let (Image width height _) = img'
            let pixels = [pixelAt img' x y | x <- [0..width-1], y <- [0..height-1]]
            return $ concatMap (\(PixelRGB8 r g b) -> [fromIntegral r / 255.0, fromIntegral g / 255.0, fromIntegral b / 255.0]) pixels

loadImages :: FilePath -> IO [(Int, [Float])]
loadImages fp = do
    -- Load the images from all the folders
    -- transform the labels into ids
    putStrLn "Loading data..."
    let labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    images <- forM (zip labels [1..]) $ \(label, idx) -> do
        let folder = fp ++ "/" ++ label
        let paths = map (\i -> folder ++ "/" ++ printf "%04d" (i :: Int) ++ ".png") [1..5000]
        images <- mapM imageToRgb paths
        return $ map (\img -> (idx, img)) images
    return $ concat images

main :: IO ()
main = do
    result <- imageToRgb "app/cifar/data/trainData/airplane/0001.png"
    print result