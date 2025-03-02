import { useRef, useState, useEffect } from 'react'
import { Box, CircularProgress } from '@mui/material'
import EditableBoundingBox from './EditableBoundingBox'
import { useImageDimensions } from '../../hooks/useImageDimensions'

/**
 * Used by AnnotationEditor and VerificationAnnotationEditor to edit the annotations of an image.
 * @param {{annotations: [{bbox: [float], class_id: Int}], image_path: String}} item: Object of annotation data for an image when called by NewAnnotations
 * @param {function} onAnnotationsChange: Callback function to update backend when any annotations changes
 * @param {[{id: Int, name: String}]} annotationCLasses: Array of annotation class id and name objects, optional
 * @param {Boolean} highlight: Whether to highlight labels based on inconsistency, optional
 * @param {function} updateRemoveLabel: Callback function when a label is removed, to update inconsistency indexes, optional 
 * @returns {JSX.Element} An image with editable annotations, which on change, 
 */
export default function EditableAnnotatedImage({ item, onAnnotationsChange, annotationClasses, highlight, updateRemoveLabel }) {
    // Set annotations to this item's annotations
    const [annotations, setAnnotations] = useState(item.annotations)

    // Image variables
    const imageRef = useRef(null)
    const containerRef = useRef(null)
    const imageDimensions = useImageDimensions(imageRef, containerRef)

    // Avoid breaks between changing images
    const [isImageLoaded, setIsImageLoaded] = useState(false)
    const previousImagePath = useRef(null)

    // Creating new annotations
    const [isDrawing, setIsDrawing] = useState(false)
    const [newAnnotation, setNewAnnotation] = useState(null)

    // Hide labels when editing
    const [isEditing, setIsEditing] = useState(false)

    // Reset annotations and loading state when item changes
    useEffect(() => {
        setAnnotations(item.annotations)
        if (item.image_path !== previousImagePath.current) {
            setIsImageLoaded(false)
            previousImagePath.current = item.image_path
            
            // TODO: not sure why if resolving 0th index image results in image loading forever
            const loadingTimer = setTimeout(() => {
                setIsImageLoaded(true)
            }, 100)
            return () => clearTimeout(loadingTimer)
        }
    }, [item])

    // Handle image loading
    function handleImageLoad() {
        setIsImageLoaded(true)
    }

    // Update annotations locally and send to parent component
    function handleUpdateAnnotation(index, updatedAnnotation) {
        const newAnnotations = [...annotations]
        newAnnotations[index] = {
            ...newAnnotations[index],
            ...updatedAnnotation
        }
        setAnnotations(newAnnotations)
        onAnnotationsChange(newAnnotations, item.image_path)
    }

    // Delete annotation locally and send to parent component
    function handleDeleteAnnotation(indexToDelete) {
        const newAnnotations = annotations.filter((_, index) => index !== indexToDelete)
        setAnnotations(newAnnotations)
        onAnnotationsChange(newAnnotations, item.image_path)
        updateRemoveLabel?.(item.image_path, indexToDelete)
    }

    // Constrain coordinates to image boundaries
    function constrainCoordinates(x, y) {
        const { width, height } = imageDimensions
        
        // Max of 0 and min is needed for negative coordinates
        const constrainedX = Math.max(0, Math.min(x, width))
        const constrainedY = Math.max(0, Math.min(y, height))

        return {
            x: constrainedX,
            y: constrainedY
        }
    }

    // Initialises drawing of a new bounding box
    function handleMouseDown(e) {
        e.preventDefault()
        e.stopPropagation()
    
        // Calculate mouse position relative to image container
        const rect = containerRef.current.getBoundingClientRect()
        const x = e.clientX - rect.left - imageDimensions.offsetX
        const y = e.clientY - rect.top - imageDimensions.offsetY

        const { x: constrainedX, y: constrainedY } = constrainCoordinates(x, y)

        setIsDrawing(true)
        setNewAnnotation({
            start: { x: constrainedX, y: constrainedY },
            end: { x: constrainedX, y: constrainedY },
            class_id: 0
        })
}

    function handleMouseMove(e) {
        if (!isDrawing) return

        e.preventDefault()
        e.stopPropagation()

        const rect = containerRef.current.getBoundingClientRect()
        const x = e.clientX - rect.left - imageDimensions.offsetX
        const y = e.clientY - rect.top - imageDimensions.offsetY

        const { x: constrainedX, y: constrainedY } = constrainCoordinates(x, y)

        setNewAnnotation(prev => ({
            ...prev,
            end: { x: constrainedX, y: constrainedY }
        }))
    }

    function handleMouseUp(e) {
        if (!isDrawing || !newAnnotation) return

        e.preventDefault()
        e.stopPropagation()
    
        // Calculate bounding box
        const { width, height } = imageDimensions

        const boxXCenter = (newAnnotation.start.x + newAnnotation.end.x) / 2
        const boxYCenter = (newAnnotation.start.y + newAnnotation.end.y) / 2
        const boxWidth = Math.abs(newAnnotation.end.x - newAnnotation.start.x)
        const boxHeight = Math.abs(newAnnotation.end.y - newAnnotation.start.y)
    
        // Normalize bounding box to range [0, 1]
        const normalizedBbox = [
            boxXCenter / width,
            boxYCenter / height,
            boxWidth / width,
            boxHeight / height,
        ]

        // Only add annotation if it has a meaningful size
        if (boxWidth > 20 && boxHeight > 20) {
            const newAnnotations = [
                ...annotations,
                {
                    bbox: normalizedBbox,
                    class_id: annotationClasses[0]?.id || 0,
                }
            ]
    
            setAnnotations(newAnnotations)
            onAnnotationsChange(newAnnotations, item.image_path)
        }
    
        setIsDrawing(false)
        setNewAnnotation(null)
    }

    return (
        <Box
            ref={containerRef}
            sx={{
                position: 'relative',
                overflow: 'hidden',
                height: '100%',
                width: '100%',
                backgroundColor: '#f3f4f6',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: isDrawing ? 'crosshair' : 'default',
                userSelect: 'none',
                'WebkitUserSelect': 'none',
                'MozUserSelect': 'none'
            }}
            onMouseLeave={handleMouseUp}
        >
            {/* Image */}
            <img
                ref={imageRef}
                src={`/api${item.image_path}`}
                alt={item.filename}
                onLoad={handleImageLoad}
                onError={handleImageLoad}
                style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    display: isImageLoaded ? 'block' : 'none',
                    userSelect: 'none',
                    'WebkitUserSelect': 'none',
                    'MozUserSelect': 'none',
                    pointerEvents: 'none'
                }}
            />
            
            {/* Loading spinner */}
            {!isImageLoaded && (
                <Box
                    sx={{
                        position: 'absolute',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        width: '100%',
                        height: '100%',
                    }}
                >
                    <CircularProgress />
                </Box>
            )}
            
            {/* Annotations */}
            {isImageLoaded && (
                <Box
                    sx={{
                        position: 'absolute',
                        zIndex: 2,
                        width: imageDimensions.width,
                        height: imageDimensions.height,
                        top: imageDimensions.offsetY,
                        left: imageDimensions.offsetX,
                    }}
                >
                    <Box
                        sx={{
                            width: '100%',
                            height: '100%',
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            cursor: 'crosshair',
                            zIndex: isDrawing ? 1000 : 'auto',

                        }}
                        onMouseDown={(e) => {
                            handleMouseDown(e)
                        }}
                        onMouseMove={(e) => {
                            handleMouseMove(e)
                        }}
                        onMouseUp={(e) => {
                            handleMouseUp(e)
                        }}
                    >
                        {/* Temporary drawing annotation */}
                        {isDrawing && newAnnotation && (
                            <Box
                                sx={{
                                    position: 'absolute',
                                    border: '2px solid blue',
                                    backgroundColor: 'rgba(0, 0, 255, 0.2)',
                                    left: Math.min(newAnnotation.start.x, newAnnotation.end.x),
                                    top: Math.min(newAnnotation.start.y, newAnnotation.end.y),
                                    width: Math.abs(newAnnotation.end.x - newAnnotation.start.x),
                                    height: Math.abs(newAnnotation.end.y - newAnnotation.start.y),
                                    pointerEvents: 'none',
                                }}
                            />
                        )}
                    </Box>
                    {annotations.map((annotation, index) => (
                        <EditableBoundingBox
                            key={index}
                            index={index}
                            class_id={annotation.class_id}
                            bbox={annotation.bbox}
                            imageDimensions={imageDimensions}
                            onUpdate={(updatedAnnotation) =>
                                handleUpdateAnnotation(index, updatedAnnotation)
                            }
                            onDelete={() => handleDeleteAnnotation(index)}
                            annotationClasses={annotationClasses}
                            highlighted={highlight && item.dataset_inconsistency_index?.includes(index)}
                            isEditing={isEditing}
                            setIsEditing={setIsEditing}
                        />
                    ))}
                </Box>
            )}
        </Box>
    )
}