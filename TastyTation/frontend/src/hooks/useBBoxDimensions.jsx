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

    useEffect(() => {
        if (imageDimensions.width === 0 || imageDimensions.height === 0) return

        const { width, height, offsetX, offsetY } = imageDimensions
        const [x_center, y_center, boxWidthNorm, boxHeightNorm] = bbox
    
        const boxWidth = boxWidthNorm * width
        const boxHeight = boxHeightNorm * height
        const boxLeft =  (x_center * width) - (boxWidth / 2)
        const boxTop = (y_center * height) - (boxHeight / 2)
    
        setStyle({
            position: 'absolute',
            display: 'block',
            width: boxWidth,
            height: boxHeight,
            x: boxLeft,
            y: boxTop
        })
    }, [bbox, imageDimensions])

    return [style, setStyle]
}