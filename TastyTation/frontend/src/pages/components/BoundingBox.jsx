import { Box } from '@mui/material'
import { useBBoxDimensions } from '../../hooks/useBBoxDimensions'

/**
 * @see EditableBoundingBox for editable version
 * @param {[Float]} bbox: Array of 4 values, representing x_center, y_center, width and height of a bounding box (YOLO format)
 * @param {{width: Int, height: Int, offsetX: Int, offsetY: Int, imageAspectRatio: Float}} imageDimensions: An object representing the dimensions of an image
 * @param {String} className: Class name of the annotation, optional
 * @param {Boolean} highlighted: Whether to highlight the box or not, optional
 * @returns {JSX.Element} Box representing a annotation, with optional label for class name and highlighted box
 */
export default function BoundingBox({ bbox, imageDimensions, className, highlighted }) {
    // Get style of bounding box
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