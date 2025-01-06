import { Box, Card, CardContent, CircularProgress, FormControl, IconButton, ImageList, ImageListItem, MenuItem, Select, Typography } from '@mui/material'
import { Delete } from '@mui/icons-material'

export default function ImagesList({ loading, images, handleDeleteImage, tiers, handleTierChange, uploadType, elevation }) {
    return (
        <Card sx={{ mb: 3 }} elevation={elevation}>
            <CardContent>
                {/* Loading Spinner */}
                {loading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                        <CircularProgress />
                    </Box>
                )}

                {/* Image List */}
                {images.size > 0 ? (
                    <ImageList sx={{ width: '100%' }} cols={3} rowHeight={300}>
                        {Array.from(images.values()).map((image) => (
                            <ImageListItem
                                key={image.name}
                                sx={{
                                    position: 'relative',
                                    border: '1px solid #eee',
                                    borderRadius: 1,
                                    overflow: 'hidden',
                                }}
                            >
                                {/* Image */}
                                <img
                                    src={image.src}
                                    alt={image.name}
                                    loading='lazy'
                                    style={{ height: '240px', objectFit: 'cover' }}
                                />
                                {/* Tier Editer */}
                                {tiers && uploadType && handleTierChange && (
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            top: 5,
                                            left: 5,
                                            p: 0.5,
                                            borderRadius: 1,
                                            cursor: 'pointer',
                                            fontSize: '0.8rem',
                                            textAlign: 'center',
                                            backgroundColor: 'rgba(255, 255, 255, 0.7)'
                                        }}
                                    >
                                        {uploadType === 'new' && (
                                            <FormControl fullWidth size='small'>
                                                <Select
                                                    value={image.tier}
                                                    onChange={(e) =>
                                                        handleTierChange(
                                                            image.path || image.name,
                                                            e.target.value
                                                        )
                                                    }
                                                >
                                                    {Object.values(tiers).map((tier) => (
                                                        <MenuItem key={tier.name} value={tier}>
                                                            {tier.name}
                                                        </MenuItem>
                                                    ))}
                                                </Select>
                                            </FormControl>
                                        )}
                                    </Box>
                                )}
                                {/* Delete button */}
                                <IconButton
                                    sx={{
                                        position: 'absolute',
                                        top: 5,
                                        right: 5,
                                        backgroundColor: 'rgba(255, 255, 255, 0.7)',
                                        '&:hover': {
                                            backgroundColor: 'rgba(255, 255, 255, 0.9)',
                                        },
                                    }}
                                    onClick={() => handleDeleteImage(image.name)}
                                >
                                    <Delete />
                                </IconButton>
                            </ImageListItem>
                        ))}
                    </ImageList>
                ) : (
                    <Typography variant='body1' color='text.secondary' align='center'>
                        No Images Uploaded
                    </Typography>
                )}
            </CardContent>
        </Card>
    )
}