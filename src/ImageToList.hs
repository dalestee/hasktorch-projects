module ImageToList (imageToRgb) where

import Codec.Picture         (readImage, pixelAt, PixelRGB8(..), DynamicImage, convertRGB8, Image(..))
import System.FilePath.Posix (splitExtension)
import Data.Word             (Word8)

imageToRgb :: FilePath -> IO [(Int, Int, Int)]
imageToRgb fp = do
    putStrLn "Loading image..."
    Right img <- readImage fp
    let (filename, ext) = splitExtension fp
    let img' = convertRGB8 img
    let (Image width height _) = img'
    let pixels = [pixelAt img' x y | x <- [0..width-1], y <- [0..height-1]]
    return $ map (\(PixelRGB8 r g b) -> (fromIntegral r, fromIntegral g, fromIntegral b)) pixels

main :: IO ()
main = do
    result <- imageToRgb "app/cifar/data/trainData/airplane/0001.png"
    print result