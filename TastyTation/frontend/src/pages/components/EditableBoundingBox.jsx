import { Rnd } from 'react-rnd'
import { useBBoxDimensions } from '../../hooks/useBBoxDimensions'

export default function EditableBoundingBox({ class_id, bbox, imageDimensions }) {
    const [style, setStyle] = useBBoxDimensions({ bbox, imageDimensions })

    if (style.display === 'none') return null

    return (
        <Rnd
            size={{ width: style.width, height: style.height }}
            position={{ x: style.x, y: style.y }}
            bounds="parent"
            onDragStop={(e, d) => {
                setStyle(prev => ({
                    ...prev,
                    x: d.x,
                    y: d.y
                }))
            }}
            onResizeStop={(e, direction, ref, delta, position) => {
                setStyle(prev => ({
                    ...prev,
                    width: parseInt(ref.style.width),
                    height: parseInt(ref.style.height),
                    ...position
                }))
            }}
            style={{
                border: '2px solid red',
                position: 'absolute',
                cursor: 'move',
            }}
        >
            <div style={{
                position: 'absolute',
                top: '0',
                left: '50%',
                transform: 'translateX(-50%)',
                color: 'red',
                backgroundColor: 'white',
                padding: '2px 5px',
                fontSize: '12px',
                fontWeight: 'bold',
                borderRadius: '3px',
                pointerEvents: 'none',
            }}>
                {class_id}
            </div>
        </Rnd>
    )
}