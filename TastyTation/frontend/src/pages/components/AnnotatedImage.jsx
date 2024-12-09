import { useRef } from 'react'
import { Box } from '@mui/material'
import BoundingBox from './BoundingBox'
import { useImageDimensions } from '../../hooks/useImageDimensions'

export default function AnnotatedImage({ item }) {
    const imageRef = useRef(null)
    const containerRef = useRef(null)
    const imageDimensions = useImageDimensions(imageRef, containerRef)

    return (
        <Box ref={containerRef} sx={{ position: 'relative', overflow: 'hidden', height: '300px', width: '100%', backgroundColor: '#f3f4f6', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <img 
                ref={imageRef} 
                src={item.image_path} 
                alt={item.filename} 
                style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
            />
            <Box sx={{ position: 'absolute', width: imageDimensions.width, height: imageDimensions.height, top: imageDimensions.offsetY, left: imageDimensions.offsetX }}>
                {item.annotations.map((annotation, index) =>
                    <BoundingBox
                        key={index}
                        bbox={annotation.bbox}
                        imageDimensions={imageDimensions}
                    />
                )}
            </Box>
        </Box>
    )
}