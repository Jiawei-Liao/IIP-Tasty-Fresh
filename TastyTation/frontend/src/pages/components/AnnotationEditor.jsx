import React from 'react'
import { Box, Button, Typography, IconButton, Paper } from '@mui/material'
import { ArrowBack, ArrowForward, Close } from '@mui/icons-material'
import AnnotatedImage from './AnnotatedImage'

export default function AnnotationEditor({ annotations, currentIndex, setCurrentIndex }) {
    const currentImage = annotations[currentIndex]

    const navigate = (direction) => {
        const newIndex = currentIndex + direction
        if (newIndex >= 0 && newIndex < annotations.length) {
            setCurrentIndex(newIndex)
        }
    }

    if (!currentImage) return null

    return (
        <Box
            sx={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                bgcolor: 'rgba(0, 0, 0, 0.7)',
                zIndex: 1200,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                p: 2,
            }}
        >
            <Paper
                sx={{
                    maxWidth: '90%',
                    maxHeight: '90%',
                    width: '800px',
                    height: 'auto',
                    display: 'flex',
                    flexDirection: 'column',
                    p: 2,
                    bgcolor: 'background.paper',
                    borderRadius: 2,
                    boxShadow: 3,
                }}
            >
                {/* Header Section */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6" noWrap>
                        {currentImage.filename}
                    </Typography>
                    <IconButton onClick={() => setCurrentIndex(-1)} aria-label="Close">
                        <Close />
                    </IconButton>
                </Box>

                {/* Annotated Image Section */}
                <Box
                    sx={{
                        flex: 1,
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        bgcolor: 'background.default',
                        p: 2,
                    }}
                >
                    <AnnotatedImage
                        item={currentImage}
                        size="large"
                        editable={true}
                    />
                </Box>

                {/* Navigation Section */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2 }}>
                    <Button
                        onClick={() => navigate(-1)}
                        disabled={currentIndex === 0}
                        startIcon={<ArrowBack />}
                        variant="outlined"
                    >
                        Previous
                    </Button>
                    <Typography>
                        Image {currentIndex + 1} of {annotations.length}
                    </Typography>
                    <Button
                        onClick={() => navigate(1)}
                        disabled={currentIndex === annotations.length - 1}
                        endIcon={<ArrowForward />}
                        variant="outlined"
                    >
                        Next
                    </Button>
                </Box>
            </Paper>
        </Box>
    )
}