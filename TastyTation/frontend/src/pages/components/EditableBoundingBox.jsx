import { useState, useRef } from 'react'
import { Box, Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField } from '@mui/material'
import { Rnd } from 'react-rnd'
import { useBBoxDimensions } from '../../hooks/useBBoxDimensions'

export default function EditableBoundingBox({ class_id, bbox, imageDimensions, onUpdate, onDelete, newAnnotationClasses }) {
    const [style, updateStyle] = useBBoxDimensions({ bbox, imageDimensions })
    const [labelDialogOpen, setLabelDialogOpen] = useState(false)
    const [isEditing, setIsEditing] = useState(false)
    const labelInputRef = useRef(null)

    if (style.display === 'none') return null

    // Update data when the bounding box is moved
    function handleDragStop(e, d) {
        setIsEditing(false)
        const updatedBBox = [
            (d.x + style.width / 2) / imageDimensions.width,
            (d.y + style.height / 2) / imageDimensions.height,
            bbox[2],
            bbox[3]
        ]

        updateStyle(updatedBBox, imageDimensions)
        onUpdate({ class_id, bbox: updatedBBox })
    }

    // Update data when the bounding box is resized
    function handleResizeStop(e, direction, ref, delta, position) {
        setIsEditing(false)
        const newWidth = parseInt(ref.style.width)
        const newHeight = parseInt(ref.style.height)
    
        const updatedBBox = [
            (position.x + newWidth / 2) / imageDimensions.width,
            (position.y + newHeight / 2) / imageDimensions.height,
            newWidth / imageDimensions.width,
            newHeight / imageDimensions.height
        ]
    
        updateStyle(updatedBBox, imageDimensions)
        onUpdate({ class_id, bbox: updatedBBox })
    }

    // Open the dialog to update the label
    function handleOpenLabelDialog() {
        setLabelDialogOpen(true)
    }

    // Update data when the label is updated and close the dialog
    function handleUpdateLabel() {
        if (labelInputRef.current) {
            const newLabel = labelInputRef.current.value
            onUpdate({ 
                class_id: newLabel, 
                bbox 
            })
            setLabelDialogOpen(false)
        }
    }

    return (
        <>
            {/* Label Display */}
            <Box 
                // Dynamically position the label above the bounding box
                ref={(el) => {
                    if (el) {
                        const labelHeight = el.offsetHeight
                        el.style.top = `${style.y - labelHeight - 5}px`
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
                    display: isEditing ? 'none' : 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    zIndex: 2,
                    whiteSpace: 'pre-wrap',
                    textAlign: 'center',
                    lineHeight: '1.2'
                }}
            >
                {newAnnotationClasses[class_id] ? newAnnotationClasses[class_id] : 'unknown'}
                <Box style={{ display: 'flex', marginTop: '4px' }}>
                    <Button 
                        size="small" 
                        onClick={handleOpenLabelDialog}
                        style={{ minWidth: 'auto', padding: '0 4px' }}
                    >
                        ✏️
                    </Button>
                    <Button 
                        size="small" 
                        onClick={onDelete}
                        style={{ minWidth: 'auto', padding: '0 4px', marginLeft: '4px' }}
                    >
                        🗑️
                    </Button>
                </Box>
            </Box>
            
            {/* Bounding Box */}
            <Rnd
                size={{ width: style.width, height: style.height }}
                position={{ x: style.x, y: style.y }}
                bounds="parent"
                onDragStart={() => setIsEditing(true)}
                onDragStop={handleDragStop}
                onResizeStart={() => setIsEditing(true)}
                onResizeStop={handleResizeStop}
                style={{
                    border: '2px solid red',
                    position: 'absolute',
                    zIndex: 1
                }}
            />

            {/* Label Edit Dialog */}
            <Dialog open={labelDialogOpen} onClose={() => setLabelDialogOpen(false)}>
                <DialogTitle>Edit Label</DialogTitle>
                <DialogContent>
                    <TextField
                        inputRef={labelInputRef}
                        autoFocus
                        margin="dense"
                        label="Class Label"
                        defaultValue={class_id}
                        fullWidth
                        variant="standard"
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setLabelDialogOpen(false)}>Cancel</Button>
                    <Button onClick={handleUpdateLabel}>Update</Button>
                </DialogActions>
            </Dialog>
        </>
    )
}