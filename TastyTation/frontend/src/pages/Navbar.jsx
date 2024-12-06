import { AppBar, Toolbar, Box, Typography, IconButton } from '@mui/material'
import { useNavigate, useLocation } from 'react-router-dom'

// Custom button component for the navbar
function NavButton({ label, isActive, to }) {
    const navigate = useNavigate()

    // Allow modified clicks to go through
    function handleClick(e) {
        if (e.ctrlKey || e.metaKey || e.shiftKey || e.altKey || e.button !== 0) {
            return
        }

        e.preventDefault()
        navigate(to)
    }

    return (
        <IconButton sx={{ borderRadius: 0 }} color="inherit" component="a" href={to} onClick={handleClick} style={{ textDecoration: 'none' }}>
            <Typography sx={{ borderBottom: isActive ? '2px solid white' : 'none', paddingBottom: isActive ? '4px' : '0'}}>
                {label}
            </Typography>
        </IconButton>
    )
}

export default function Navbar() {
    // Helper function to check if the current path matches the route
    const location = useLocation()
    function isActive(path) {
        return location.pathname === path
    }

    return (
        <Box sx={{ mb: 3 }}>
            <AppBar position="static">
                <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', gap: 2, pl: 2 }}>
                        <Typography variant="h6">
                            TastyTation
                        </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 2, pr: 2 }}>
                        <NavButton label="Upload Data" isActive={isActive('/upload')} to='/upload' />
                        <NavButton label="New Annotations" isActive={isActive('/new-annotations')} to='/new-annotations' />
                        <NavButton label="Classifiers" isActive={isActive('/classifiers')} to='/classifiers' />
                        <NavButton label="Verify Dataset" isActive={isActive('/verify')} to='/verify' />
                        <NavButton label="Train" isActive={isActive('/train')} to='/train' />
                    </Box>
                </Toolbar>
            </AppBar>
        </Box>
    )
}
