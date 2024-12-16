import { useState, useEffect } from 'react'

/**
 * Helper hook to calculate the style of a Box div based on the bounding box and image dimensions
 * @param {[Float]} bbox: Array of 4 values, representing x_center, y_center, width and height of a bounding box (YOLO format)
 * @param {{width: Int, height: Int, offsetX: Int, offsetY: Int, imageAspectRatio: Float}} imageDimensions: An object representing the dimensions of an image
 * @returns {{position: String, display: String, width: Int, height: Int, x: int, y: Int}, function} Calculated style and update function for the bounding box
 */
export function useBBoxDimensions({ bbox, imageDimensions }) {
    const [style, setStyle] = useState({
        position: 'absolute',
        display: 'none',
        width: 0,
        height: 0,
        x: 0,
        y: 0
    })
    console.log(imageDimensions)
    function calculateStyle(bbox, imageDimensions) {
        const { width, height } = imageDimensions
        const [x_center, y_center, boxWidthNorm, boxHeightNorm] = bbox
    
        const boxWidth = boxWidthNorm * width
        const boxHeight = boxHeightNorm * height
        const boxLeft =  (x_center * width) - (boxWidth / 2)
        const boxTop = (y_center * height) - (boxHeight / 2)
    
        return {
            position: 'absolute',
            display: 'block',
            width: boxWidth,
            height: boxHeight,
            x: boxLeft,
            y: boxTop
        }
    }
        
    useEffect(() => {
        if (imageDimensions.width === 0 || imageDimensions.height === 0) return
        setStyle(calculateStyle(bbox, imageDimensions))
    }, [bbox, imageDimensions])

    function updateStyle(bbox, imageDimensions) {
        setStyle(calculateStyle(bbox, imageDimensions))
    }

    return [style, updateStyle]
}