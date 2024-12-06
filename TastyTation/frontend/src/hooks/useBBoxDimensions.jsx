import { useState, useEffect } from 'react'

export function useBBoxDimensions({ bbox, imageDimensions }) {
    const [style, setStyle] = useState({
        position: 'absolute',
        display: 'none',
        width: 0,
        height: 0,
        x: 0,
        y: 0
    })

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