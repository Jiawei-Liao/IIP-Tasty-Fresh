import { Box } from '@mui/material'
import { useBBoxDimensions } from '../../hooks/useBBoxDimensions'

export default function BoundingBox({ bbox, imageDimensions, className, highlighted }) {
    const [style] = useBBoxDimensions({ bbox, imageDimensions })

    return (
        <>
            <Box 
                // Dynamically position the label above the bounding box
                ref={(el) => {
                    if (el) {
                        const labelHeight = el.offsetHeight
                        let topPosition = style.y - labelHeight - 5
            
                        // Clip label to top of image if it goes off screen
                        if (topPosition < 0) {
                            topPosition = 10
                        }
            
                        el.style.top = `${topPosition}px`
                    }
                }}
                style={{
                    position: 'absolute',
                    left: style.x + style.width / 2,
                    transform: 'translateX(-50%)',
                    color: 'red',
                    backgroundColor: 'white',
                    padding: '2px 5px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    borderRadius: '3px',
                    display: className? 'flex' : 'none',
                    flexDirection: 'column',
                    alignItems: 'center',
                    zIndex: 2,
                    whiteSpace: 'pre-wrap',
                    textAlign: 'center',
                    lineHeight: '1.2'
                }}
            >
                {className}
            </Box>
            <Box 
                style={{
                    ...style,
                    left: style.x,
                    top: style.y,
                    border: '2px solid red',
                    backgroundColor: highlighted ? 'rgba(255, 0, 0, 0.5)' : 'transparent'
                }} 
            />
        </>
    )
}