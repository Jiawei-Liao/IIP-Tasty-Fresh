import React, { useState } from 'react'
import { Box, Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField, List, ListItem, ListItemText } from '@mui/material'
import { Rnd } from 'react-rnd'
import { useBBoxDimensions } from '../../hooks/useBBoxDimensions'

/**
 * @see BoundingBox for not editable version
 * @param {Int} class_id: ID of the class this bounding box represents
 * @param {[Float]} bbox: Array of 4 values, representing x_center, y_center, width and height of a bounding box (YOLO format)
 * @param {{width: Int, height: Int, offsetX: Int, offsetY: Int, imageAspectRatio: Float}} imageDimensions: An object representing the dimensions of an image
 * @param {function} onUpdate: Callback function to update annotations when this bounding box changes
 * @param {function} onDelete: Callback function to delete this annotation
 * @param {[{id: Int, name: String}]} annotationCLasses: Array of annotation class id and name objects
 * @param {Boolean} highlighted: Whether to highlight this bounding box or not
 * @param {Boolean} isEditing: Whether to hide labels or not
 * @param {React.Dispatch<Boolean>} setIsEditing: Set whether is editing or not
 * @returns {JSX.Element} Rnd editable box with label for annotation class name and edit and delete functionality
 */
export default function EditableBoundingBox({ class_id, bbox, imageDimensions, onUpdate, onDelete, annotationClasses, highlighted, isEditing, setIsEditing }) {
    // Get style of bounding box
    const [style, updateStyle] = useBBoxDimensions({ bbox, imageDimensions })

    // Variables for editing labels
    const [labelDialogOpen, setLabelDialogOpen] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const [filteredClasses, setFilteredClasses] = useState(annotationClasses)

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

    // Filter the class names based on the search query
    function handleSearchChange(event) {
        const query = event.target.value
        setSearchQuery(query)
        setFilteredClasses(
            annotationClasses.filter(classItem =>
                classItem.name.toLowerCase().includes(query.toLowerCase())
            )
        )
    }

    // Update data when the label is selected and close the dialog
    function handleSelectLabel(id) {
        onUpdate({ class_id: id, bbox })
        setLabelDialogOpen(false)
    }

    const currentClassName = annotationClasses.find(item => item.id === class_id)?.name || 'unknown'

    return (
        <>
            {/* Label Display */}
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
                    display: isEditing ? 'none' : 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    zIndex: 2,
                    whiteSpace: 'pre-wrap',
                    textAlign: 'center',
                    lineHeight: '1.2'
                }}
            >
                {currentClassName}
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
                    zIndex: 1,
                    backgroundColor: highlighted ? 'rgba(255, 0, 0, 0.5)' : 'transparent'
                }}
            />

            {/* Label Edit Dialog */}
            <Dialog open={labelDialogOpen} onClose={() => setLabelDialogOpen(false)} fullWidth maxWidth='sm'>
                <DialogTitle>Edit Label</DialogTitle>
                <DialogContent>
                    {/* Search Input */}
                    <TextField
                        value={searchQuery}
                        onChange={handleSearchChange}
                        autoFocus
                        margin="dense"
                        label="Search Class Label"
                        fullWidth
                        variant="standard"
                    />
                    <List>
                        {filteredClasses.map(({ id, name }) => (
                            <ListItem button key={id} onClick={() => handleSelectLabel(id)}>
                                <ListItemText primary={name} />
                            </ListItem>
                        ))}
                    </List>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setLabelDialogOpen(false)}>Cancel</Button>
                </DialogActions>
            </Dialog>
        </>
    )
}
