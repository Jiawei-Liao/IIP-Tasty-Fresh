import { Box } from '@mui/material'
import { useBBoxDimensions } from '../../hooks/useBBoxDimensions'

export default function BoundingBox({ bbox, imageDimensions }) {
    const [style] = useBBoxDimensions({ bbox, imageDimensions })

    return (
        <Box 
            style={{
                ...style,
                left: style.x,
                top: style.y,
                border: '2px solid red'
            }} 
        />
    )
}