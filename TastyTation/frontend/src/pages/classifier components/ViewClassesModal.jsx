import { Box, Stack, Tooltip, Dialog, DialogActions, DialogContent, DialogTitle, Button, Typography } from '@mui/material'

const trainColour = '#4caf50'
const valColour = '#ff9800'
const testColour = '#f44336'

function SplitBar({ train = 0, val = 0, test = 0, total = 0 }) {
    const width1 = (train / total) * 100
    const width2 = (val / total) * 100
    const width3 = (test / total) * 100


    if (total === 0) {
        return (
            <Box sx={{ width: '100%', height: 20, display: 'flex', borderRadius: 2, overflow: 'hidden' }}>
                <Tooltip title={`No Images`} placement='top'>
                    <Box
                        sx={{
                            width: `100%`,
                            backgroundColor: 'grey', 
                            cursor: 'pointer'
                        }}
                    />
                </Tooltip>
            </Box>
        )
    }

    return (
        <Box sx={{ width: '100%', height: 20, display: 'flex', borderRadius: 2, overflow: 'hidden' }}>
            <Tooltip title={`Train: ${train}`} placement='top'>
                <Box
                    sx={{
                        width: `${width1}%`,
                        backgroundColor: trainColour, 
                        cursor: 'pointer'
                    }}
                />
            </Tooltip>

            <Tooltip title={`Valid: ${val}`} placement='top'>
                <Box
                    sx={{
                        width: `${width2}%`,
                        backgroundColor: valColour,
                        cursor: 'pointer',
                    }}
                />
            </Tooltip>

            <Tooltip title={`Test: ${test}`} placement='top'>
                <Box
                    sx={{
                        width: `${width3}%`,
                        backgroundColor: testColour,
                        cursor: 'pointer',
                    }}
                />
            </Tooltip>
        </Box>
    )
}

export default function ViewClassesModel({ classes, setClasses, classifierName, setError }) {
    function handleClose() {
        setClasses({})
    }

    const sortedClasses = Object.keys(classes)
    .sort((a, b) => classes[b].total - classes[a].total)
    .reduce((acc, key) => {
        acc[key] = classes[key]
        return acc
    }, {})

    function handleRename(key) {
        const newName = prompt(`Enter a new name for the class '${key}'`)
        if (newName) {
            const formData = new FormData()
            formData.append('classifierName', classifierName)
            formData.append('oldClassName', key)
            formData.append('newClassName', newName)
            
            fetch('/api/rename-class', {
                method: 'POST',
                body: formData
            })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error('Failed to rename class')
                    }
                    return response.json()
                })
                .then((data) => {
                    if (data && data.classes && Object.keys(data.classes).length > 0) {
                        setClasses(data.classes)
                    } else {
                        setError('No classes found')
                    }
                })
                .catch((error) => {
                    setError(error.message)
                })
        }
    }

    function handleDelete(key) {
        const confirm = window.confirm(`Are you sure you want to delete the class '${key}'?`)
        if (confirm) {
            const formData = new FormData()
            formData.append('classifierName', classifierName)
            formData.append('className', key)

            fetch('/api/delete-class', {
                method: 'POST',
                body: formData
            })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error('Failed to delete class')
                    }
                    return response.json()
                })
                .then((data) => {
                    if (data && data.classes && Object.keys(data.classes).length > 0) {
                        setClasses(data.classes)
                    } else {
                        setError('No classes found')
                    }
                })
                .catch((error) => {
                    setError(error.message)
                })
        }
    }

    return (
        <Dialog open={Object.keys(classes).length > 0} onClose={handleClose} fullWidth sx={{ width: '800px', margin: 'auto' }}>
            <DialogTitle>Classes Overview</DialogTitle>
            <DialogContent>
                <Stack spacing={2}>
                    {Object.keys(sortedClasses).map((key) => {
                        const { train, val, test, total } = sortedClasses[key]
                        return (
                            <Box key={key}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Typography variant='subtitle1' sx={{ fontWeight: 'bold' }}>
                                        {key} <span style={{ fontWeight: 'normal', color: 'gray' }}>({total})</span>
                                    </Typography>
                                    <Box sx={{ display: 'flex', gap: 1 }}>
                                        <Button variant='outlined' color='primary' size='small' onClick={() => handleRename(key)}>
                                            Rename
                                        </Button>
                                        <Button variant='outlined' color='error' size='small' onClick={() => handleDelete(key)}>
                                            Delete
                                        </Button>
                                    </Box>
                                </Box>
                                {/* Adding a gap between text/buttons and the split bar */}
                                <Box sx={{ marginTop: 1 }}>
                                    <SplitBar train={train} val={val} test={test} total={total} />
                                </Box>
                            </Box>
                        )
                    })}
                </Stack>
            </DialogContent>
            <DialogActions>
                <Button onClick={handleClose} color='primary'>
                    Close
                </Button>
            </DialogActions>
        </Dialog>
    )
}