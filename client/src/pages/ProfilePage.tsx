import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { LogOut, User, X } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import { useNavigate } from 'react-router-dom';

export const ProfilePage = () => {
  const { user, signOut } = useAuth();
  const [isSigningOut, setIsSigningOut] = useState(false);
  const navigate = useNavigate();

  const handleBack = () => {
    navigate('/');
  };

  // Handle Escape key press
  useEffect(() => {
    const handleEscKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        handleBack();
      }
    };

    window.addEventListener('keydown', handleEscKey);
    
    return () => {
      window.removeEventListener('keydown', handleEscKey);
    };
  }, []);

  const handleSignOut = async () => {
    try {
      setIsSigningOut(true);
      await signOut();
    } catch (error) {
      toast({
        title: "Sign out failed",
        description: "There was a problem signing you out.",
        variant: "destructive",
      });
    } finally {
      setIsSigningOut(false);
    }
  };

  // Get user initials for avatar fallback
  const userInitials = user?.email 
    ? user.email.charAt(0).toUpperCase() 
    : 'U';

  return (
    <div className="flex flex-col items-center justify-center p-6 min-h-screen bg-gray-900 relative">
      {/* Back button */}
      <button 
        onClick={handleBack}
        className="absolute top-6 left-6 p-2 rounded-full bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
        aria-label="Go back"
      >
        <X className="w-5 h-5" />
      </button>
      
      <Card className="w-full max-w-md bg-gray-800 border-gray-700">
        <CardHeader className="flex flex-col items-center">
          <Avatar className="h-24 w-24 mb-4">
            <AvatarImage src={user?.user_metadata?.avatar_url} alt="Profile picture" />
            <AvatarFallback className="bg-blue-600 text-white text-xl">
              {userInitials}
            </AvatarFallback>
          </Avatar>
          <CardTitle className="text-2xl text-white">Profile</CardTitle>
          <CardDescription className="text-gray-400">
            Your account information
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-400">Email</h3>
            <p className="text-white">{user?.email || 'No email available'}</p>
          </div>
          
          {user?.user_metadata?.full_name && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-400">Full Name</h3>
              <p className="text-white">{user.user_metadata.full_name}</p>
            </div>
          )}
          
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-400">Account Type</h3>
            <p className="text-white">
              {user?.app_metadata?.provider || 'Email'}
            </p>
          </div>
          
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-400">Member Since</h3>
            <p className="text-white">
              {user?.created_at 
                ? new Date(user.created_at).toLocaleDateString() 
                : 'Unknown'}
            </p>
          </div>
        </CardContent>
        <CardFooter>
          <Button 
            onClick={handleSignOut}
            disabled={isSigningOut}
            className="w-full bg-red-600 hover:bg-red-700 text-white"
          >
            {isSigningOut ? "Signing out..." : "Sign Out"}
            <LogOut className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default ProfilePage; 